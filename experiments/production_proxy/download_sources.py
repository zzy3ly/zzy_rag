"""Download public FORCE text exports with provenance and checksum verification."""
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
response = requests.get("https://zenodo.org/api/records/10775273", timeout=40)
response.raise_for_status()
record = response.json()
(RAW / "zenodo_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
for name, local in [
    ("Norway - Norwegian Petroleum Directorate relinquishment reports.csv", "norway_reports.csv"),
    ("UK - North Sea Transition Authority NDR reports.csv", "uk_reports.csv"),
]:
    entry = next(f for f in record["files"] if f["key"] == name)
    path = RAW / local
    if not path.exists():
        temporary = path.with_suffix(".partial")
        with requests.get(entry["links"]["self"], stream=True, timeout=(30, 90)) as r:
            r.raise_for_status()
            with temporary.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
        temporary.replace(path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if f"md5:{digest}" != entry["checksum"]:
        raise RuntimeError(f"Checksum mismatch: {local}")
    print(f"VERIFIED {local} {path.stat().st_size} bytes", flush=True)
