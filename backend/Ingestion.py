"""Persistent asynchronous document ingestion and version lifecycle management.

The default worker is deliberately dependency-light so the repository runs on a
single machine.  ``IngestionManager`` is the queue boundary: a production
deployment can replace its enqueue method with Celery/Redis without changing the
state machine, idempotency rules, or vector activation protocol.
"""

from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
import uuid

from StructureChunking import PIPELINE_VERSION, load_minilm_tokenizer, structure_parent_child_chunks
from UploadHandler import UploadHandler


TERMINAL = {"ACTIVE", "FAILED"}
RUNNING = {"PARSING", "CHUNKING", "EMBEDDING", "INDEXING", "VALIDATING", "ACTIVATING"}


def _now():
    return datetime.now(timezone.utc).isoformat()


class IngestionStore:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self):
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS managed_documents (
              document_id TEXT PRIMARY KEY,
              collection_id TEXT NOT NULL,
              document_key TEXT NOT NULL,
              active_version_id TEXT,
              created_at TEXT NOT NULL,
              UNIQUE(collection_id, document_key)
            );
            CREATE TABLE IF NOT EXISTS document_versions (
              version_id TEXT PRIMARY KEY,
              document_id TEXT NOT NULL REFERENCES managed_documents(document_id),
              version_no INTEGER NOT NULL,
              content_sha256 TEXT NOT NULL,
              filename TEXT NOT NULL,
              blob_path TEXT NOT NULL,
              pipeline_version TEXT NOT NULL,
              status TEXT NOT NULL,
              parent_count INTEGER DEFAULT 0,
              chunk_count INTEGER DEFAULT 0,
              error TEXT,
              created_at TEXT NOT NULL,
              activated_at TEXT,
              UNIQUE(document_id, content_sha256, pipeline_version),
              UNIQUE(document_id, version_no)
            );
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
              job_id TEXT PRIMARY KEY,
              version_id TEXT NOT NULL REFERENCES document_versions(version_id),
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL,
              error TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingestion_audit (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              document_id TEXT NOT NULL,
              version_id TEXT,
              job_id TEXT,
              event TEXT NOT NULL,
              details TEXT,
              created_at TEXT NOT NULL
            );
            """)

    @staticmethod
    def row(row):
        return dict(row) if row else None

    def submit(self, collection_id, document_key, filename, content_sha256,
               blob_path, pipeline_version, max_attempts):
        now = _now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = db.execute(
                "SELECT * FROM managed_documents WHERE collection_id=? AND document_key=?",
                (collection_id, document_key)).fetchone()
            if document is None:
                document_id = str(uuid.uuid4())
                db.execute("INSERT INTO managed_documents VALUES (?, ?, ?, NULL, ?)",
                           (document_id, collection_id, document_key, now))
            else:
                document_id = document["document_id"]
            duplicate = db.execute("""
                SELECT v.*, j.job_id, j.status AS job_status
                FROM document_versions v JOIN ingestion_jobs j USING(version_id)
                WHERE v.document_id=? AND v.content_sha256=? AND v.pipeline_version=?
                ORDER BY j.created_at DESC LIMIT 1
            """, (document_id, content_sha256, pipeline_version)).fetchone()
            if duplicate:
                db.commit()
                result = dict(duplicate)
                result["deduplicated"] = True
                return result
            version_no = db.execute(
                "SELECT COALESCE(MAX(version_no), 0) + 1 FROM document_versions WHERE document_id=?",
                (document_id,)).fetchone()[0]
            version_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
            db.execute("""INSERT INTO document_versions
                (version_id, document_id, version_no, content_sha256, filename,
                 blob_path, pipeline_version, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)""",
                (version_id, document_id, version_no, content_sha256, filename,
                 str(blob_path), pipeline_version, now))
            db.execute("""INSERT INTO ingestion_jobs
                (job_id, version_id, status, max_attempts, created_at, updated_at)
                VALUES (?, ?, 'PENDING', ?, ?, ?)""",
                (job_id, version_id, max_attempts, now, now))
            self._audit(db, document_id, version_id, job_id, "SUBMITTED",
                        {"filename": filename, "sha256": content_sha256})
            db.commit()
            return {"document_id": document_id, "version_id": version_id,
                    "version_no": version_no, "job_id": job_id,
                    "job_status": "PENDING", "deduplicated": False}

    def _audit(self, db, document_id, version_id, job_id, event, details=None):
        db.execute("""INSERT INTO ingestion_audit
            (document_id, version_id, job_id, event, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (document_id, version_id, job_id, event,
             json.dumps(details or {}, ensure_ascii=False), _now()))

    def job(self, job_id):
        with self.connect() as db:
            row = db.execute("""SELECT j.*, v.document_id, v.version_no,
                v.filename, v.content_sha256, v.pipeline_version,
                v.parent_count, v.chunk_count, v.activated_at
                FROM ingestion_jobs j JOIN document_versions v USING(version_id)
                WHERE j.job_id=?""", (job_id,)).fetchone()
            return self.row(row)

    def work_item(self, job_id):
        with self.connect() as db:
            return self.row(db.execute("""SELECT j.*, v.document_id, v.filename,
                v.blob_path, v.pipeline_version, d.collection_id, d.document_key
                FROM ingestion_jobs j JOIN document_versions v USING(version_id)
                JOIN managed_documents d USING(document_id) WHERE j.job_id=?""",
                (job_id,)).fetchone())

    def set_state(self, job_id, status, error=None, counts=None, increment=False):
        counts = counts or {}
        with self.connect() as db:
            item = db.execute("""SELECT j.version_id, v.document_id
                FROM ingestion_jobs j JOIN document_versions v USING(version_id)
                WHERE j.job_id=?""", (job_id,)).fetchone()
            if not item:
                raise KeyError(job_id)
            db.execute("""UPDATE ingestion_jobs SET status=?, error=?, updated_at=?,
                attempts=attempts+? WHERE job_id=?""",
                (status, error, _now(), 1 if increment else 0, job_id))
            db.execute("""UPDATE document_versions SET status=?, error=?,
                parent_count=COALESCE(?, parent_count), chunk_count=COALESCE(?, chunk_count)
                WHERE version_id=?""",
                (status, error, counts.get("parents"), counts.get("chunks"), item["version_id"]))
            self._audit(db, item["document_id"], item["version_id"], job_id,
                        status, {"error": error} if error else counts)

    def complete_activation(self, job_id):
        now = _now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            item = db.execute("""SELECT j.version_id, v.document_id
                FROM ingestion_jobs j JOIN document_versions v USING(version_id)
                WHERE j.job_id=?""", (job_id,)).fetchone()
            previous = db.execute(
                "SELECT active_version_id FROM managed_documents WHERE document_id=?",
                (item["document_id"],)).fetchone()[0]
            if previous and previous != item["version_id"]:
                db.execute("UPDATE document_versions SET status='ARCHIVED' WHERE version_id=?",
                           (previous,))
            db.execute("""UPDATE document_versions SET status='ACTIVE', activated_at=?, error=NULL
                WHERE version_id=?""", (now, item["version_id"]))
            db.execute("UPDATE managed_documents SET active_version_id=? WHERE document_id=?",
                       (item["version_id"], item["document_id"]))
            db.execute("""UPDATE ingestion_jobs SET status='ACTIVE', error=NULL, updated_at=?
                WHERE job_id=?""", (now, job_id))
            self._audit(db, item["document_id"], item["version_id"], job_id,
                        "ACTIVE", {"replaced_version_id": previous})

    def recoverable_jobs(self):
        with self.connect() as db:
            placeholders = ",".join("?" for _ in RUNNING)
            db.execute(f"UPDATE ingestion_jobs SET status='PENDING', updated_at=? WHERE status IN ({placeholders})",
                       (_now(), *RUNNING))
            db.execute(f"UPDATE document_versions SET status='PENDING' WHERE status IN ({placeholders})",
                       tuple(RUNNING))
            return [row[0] for row in db.execute(
                "SELECT job_id FROM ingestion_jobs WHERE status IN ('PENDING','RETRYING')")]

    def retry(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT status FROM ingestion_jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise KeyError(job_id)
            if row[0] != "FAILED":
                raise ValueError("only FAILED jobs can be retried")
            db.execute("UPDATE ingestion_jobs SET status='PENDING', error=NULL, updated_at=? WHERE job_id=?",
                       (_now(), job_id))
            db.execute("""UPDATE document_versions SET status='PENDING', error=NULL
                WHERE version_id=(SELECT version_id FROM ingestion_jobs WHERE job_id=?)""", (job_id,))

    def version(self, version_id):
        with self.connect() as db:
            return self.row(db.execute("""SELECT v.*, d.collection_id, d.document_key,
                d.active_version_id FROM document_versions v JOIN managed_documents d USING(document_id)
                WHERE version_id=?""", (version_id,)).fetchone())

    def rollback(self, document_id, version_id):
        now = _now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            target = db.execute("SELECT * FROM document_versions WHERE document_id=? AND version_id=?",
                                (document_id, version_id)).fetchone()
            if not target or target["status"] not in ("ACTIVE", "ARCHIVED"):
                raise ValueError("target version is not an activatable version")
            previous = db.execute("SELECT active_version_id FROM managed_documents WHERE document_id=?",
                                  (document_id,)).fetchone()[0]
            if previous and previous != version_id:
                db.execute("UPDATE document_versions SET status='ARCHIVED' WHERE version_id=?", (previous,))
            db.execute("UPDATE document_versions SET status='ACTIVE', activated_at=? WHERE version_id=?",
                       (now, version_id))
            db.execute("UPDATE managed_documents SET active_version_id=? WHERE document_id=?",
                       (version_id, document_id))
            self._audit(db, document_id, version_id, None, "ROLLBACK",
                        {"replaced_version_id": previous})

    def history(self, document_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute("""SELECT version_id, version_no,
                filename, content_sha256, pipeline_version, status, parent_count,
                chunk_count, created_at, activated_at, error
                FROM document_versions WHERE document_id=? ORDER BY version_no DESC""",
                (document_id,))]


class IngestionManager:
    def __init__(self, embedding_handler, database_path="ingestion_data/lifecycle.db",
                 blob_directory="ingestion_data/blobs", tokenizer=None,
                 workers=1, max_attempts=3, start_workers=True):
        self.embedding_handler = embedding_handler
        self.store = IngestionStore(database_path)
        self.blob_directory = Path(blob_directory)
        self.blob_directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.upload_handler = UploadHandler()
        self.max_attempts = max_attempts
        self.queue = queue.Queue()
        self._threads = []
        if start_workers:
            for number in range(workers):
                thread = threading.Thread(target=self._worker,
                    name=f"ingestion-worker-{number}", daemon=True)
                thread.start()
                self._threads.append(thread)
        for job_id in self.store.recoverable_jobs():
            self.queue.put(job_id)

    def submit(self, collection_id, filename, content, document_key=None):
        if not content:
            raise ValueError("empty files cannot be ingested")
        digest = hashlib.sha256(content).hexdigest()
        blob = self.blob_directory / f"{digest}.bin"
        if not blob.exists():
            temporary = blob.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(content)
            temporary.replace(blob)
        return self._register_blob(
            collection_id, filename, document_key, digest, blob)

    def submit_stream(self, collection_id, filename, stream, document_key=None,
                      buffer_size=1024 * 1024):
        """Hash and persist an upload without holding the whole file in memory."""
        temporary = self.blob_directory / f"upload-{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                while True:
                    block = stream.read(buffer_size)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size == 0:
                raise ValueError("empty files cannot be ingested")
            blob = self.blob_directory / f"{digest.hexdigest()}.bin"
            if blob.exists():
                temporary.unlink()
            else:
                temporary.replace(blob)
            return self._register_blob(
                collection_id, filename, document_key, digest.hexdigest(), blob)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _register_blob(self, collection_id, filename, document_key, digest, blob):
        result = self.store.submit(
            collection_id, document_key or filename, filename, digest, blob,
            PIPELINE_VERSION, self.max_attempts)
        if not result["deduplicated"]:
            self.queue.put(result["job_id"])
        return result

    def _worker(self):
        while True:
            job_id = self.queue.get()
            try:
                if job_id is None:
                    return
                self.process(job_id)
            finally:
                self.queue.task_done()

    def shutdown(self, timeout=2):
        for _ in self._threads:
            self.queue.put(None)
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()

    def process(self, job_id):
        item = self.store.work_item(job_id)
        if not item or item["status"] in TERMINAL:
            return
        version_id = item["version_id"]
        previous_active = None
        activated = False
        try:
            # Deterministic chunk IDs make retry safe after a process crash.
            # Restore the DB-recorded active version and discard a partial target.
            recorded_active = self.store.version(version_id)["active_version_id"]
            if recorded_active and recorded_active != version_id:
                self.embedding_handler.activate_version(
                    item["document_id"], recorded_active)
            try:
                self.embedding_handler.remove_version(version_id)
            except RuntimeError:
                self.embedding_handler.deactivate_version(version_id)
                self.embedding_handler.remove_version(version_id)
            self.store.set_state(job_id, "PARSING", increment=True)
            parsed = self.upload_handler.extract_text(
                Path(item["blob_path"]).read_bytes(), item["filename"])
            if parsed[0] != "success":
                raise ValueError(parsed[1])
            self.store.set_state(job_id, "CHUNKING")
            if self.tokenizer is None:
                encoder = getattr(self.embedding_handler, "encoder", None)
                self.tokenizer = (encoder.tokenizer if encoder is not None
                                  else load_minilm_tokenizer())
            result = structure_parent_child_chunks(
                parsed[1], self.tokenizer, item["filename"])
            if not result["children"]:
                raise ValueError("document produced no indexable chunks")
            counts = {"parents": len(result["parents"]),
                      "chunks": len(result["children"])}
            self.store.set_state(job_id, "EMBEDDING", counts=counts)
            self.embedding_handler.stage_version(
                result["children"], item["document_id"], version_id,
                item["collection_id"], item["filename"])
            self.store.set_state(job_id, "VALIDATING", counts=counts)
            self.embedding_handler.validate_version(version_id, counts["chunks"])
            self.store.set_state(job_id, "ACTIVATING", counts=counts)
            previous_active = self.store.version(version_id)["active_version_id"]
            self.embedding_handler.activate_version(item["document_id"], version_id)
            activated = True
            self.store.complete_activation(job_id)
        except Exception as exc:
            if activated and previous_active:
                try:
                    self.embedding_handler.activate_version(
                        item["document_id"], previous_active)
                    activated = False
                except Exception:
                    pass
            elif activated:
                try:
                    self.embedding_handler.deactivate_version(version_id)
                    activated = False
                except Exception:
                    pass
            try:
                if not activated:
                    self.embedding_handler.remove_version(version_id)
            except Exception:
                pass
            current = self.store.job(job_id)
            message = f"{type(exc).__name__}: {str(exc)}"[:1000]
            if current and current["attempts"] < current["max_attempts"]:
                self.store.set_state(job_id, "RETRYING", error=message)
                self.store.set_state(job_id, "PENDING", error=message)
                self.queue.put(job_id)
            else:
                self.store.set_state(job_id, "FAILED", error=message)

    def status(self, job_id):
        return self.store.job(job_id)

    def retry(self, job_id):
        self.store.retry(job_id)
        self.queue.put(job_id)
        return self.status(job_id)

    def rollback(self, document_id, version_id):
        target = self.store.version(version_id)
        if not target or target["document_id"] != document_id:
            raise KeyError(version_id)
        previous = target["active_version_id"]
        self.embedding_handler.activate_version(document_id, version_id)
        try:
            self.store.rollback(document_id, version_id)
        except Exception:
            if previous:
                self.embedding_handler.activate_version(document_id, previous)
            raise
        return self.store.version(version_id)

    def wait(self, job_id, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.status(job_id)
            if result and result["status"] in TERMINAL:
                return result
            time.sleep(0.02)
        raise TimeoutError(job_id)
