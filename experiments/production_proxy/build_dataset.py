"""Freeze independently authored public evidence and explicit synthetic fixtures."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    selected = json.loads((ROOT / "selection.json").read_text())
    rows = list(csv.DictReader((ROOT / "raw/norway_reports.csv").open(encoding="utf-8-sig", newline="")))
    groups = defaultdict(list)
    for row in rows:
        groups[row["filename"]].append(row)
    # Check the deterministic selection, before observing retrieval.
    eligible = [n for n, rs in groups.items()
                if 8 <= len(rs) <= 30 and 15000 <= sum(len(r["content"]) for r in rs) <= 65000]
    eligible.sort(key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    assert selected == eligible[:40]
    documents, lookup, audits = [], {}, []
    for i, name in enumerate(selected):
        rs = sorted(groups[name], key=lambda r: int(r["page"]))
        pages = [int(r["page"]) for r in rs]
        assert len(pages) == len(set(pages)), f"Duplicate page: {name}"
        family = f"public-{i:02d}"
        for r in rs:
            doc = {"id": f"{family}-p{int(r['page']):03d}", "family": family,
                   "title": name, "page": int(r["page"]), "text": r["content"],
                   "source_type": "public_ocr", "source_record": r["_id"],
                   "source_url": "https://doi.org/10.5281/zenodo.10775273",
                   "roles": ["public"], "valid_from": "0001-01-01", "valid_to": "9999-12-31"}
            documents.append(doc)
            lookup[i, int(r["page"])] = doc
        audits.append({"filename": name, "family": family, "exported_pages": pages,
                       "missing_between_first_last": sorted(set(range(min(pages), max(pages)+1)) - set(pages)),
                       "characters": sum(len(r["content"]) for r in rs),
                       "question_source": i < 20})
    facts = list(csv.DictReader((ROOT / "facts.tsv").open(encoding="utf-8"), delimiter="\t"))
    questions, by_family = [], defaultdict(list)
    for i, fact in enumerate(facts):
        doc = lookup[int(fact["doc"]), int(fact["page"])]
        quote = fact["evidence"]
        assert quote in doc["text"], f"Evidence not found: fact {i}: {quote}"
        # Alternatives include exact duplicate evidence elsewhere in SAME report.
        alternatives = []
        for page in documents:
            if page["family"] != doc["family"]:
                continue
            start = 0
            while (start := page["text"].find(quote, start)) >= 0:
                alternatives.append({"document_id": page["id"], "start": start, "end": start+len(quote)})
                start += len(quote)
        evidence = [alternatives]
        split = "dev" if int(fact["doc"]) % 5 == 0 else "test"
        for variant, key in [("direct", "question"), ("paraphrase", "paraphrase")]:
            q = {"id": f"public-f{i:03d}-{variant}", "family": doc["family"], "split": split,
                 "query": fact[key], "category": variant, "source_type": "public_ocr",
                 "expected_action": "retrieve", "evidence_groups": evidence,
                 "reference_evidence": [quote], "context": {"roles": ["public"], "as_of": "2026-01-01"},
                 "label_status": "AI-authored and source-checked; not expert adjudicated"}
            questions.append(q)
        by_family[doc["family"]].append(questions[-2])
    for family, fs in by_family.items():
        assert len(fs) == 3
        # An explicitly two-part request, not a claim of complex reasoning.
        q = dict(fs[0])
        q.update(id=family+"-multi", category="multi_evidence",
                 query="For a review memo, address both points with supporting sources: " +
                       fs[0]["query"] + " Also, " + fs[1]["query"],
                 evidence_groups=fs[0]["evidence_groups"] + fs[1]["evidence_groups"],
                 reference_evidence=fs[0]["reference_evidence"] + fs[1]["reference_evidence"])
        questions.append(q)
    # Administrative mock records, NOT real petroleum operating instructions.
    for i in range(10):
        family = f"synthetic-{i:02d}"
        asset = f"PX-{410+i}A"
        site = ["North", "West", "East", "South", "Central"][i % 5] + f"-{i//5+1}"
        split = "dev" if i % 5 == 0 else "test"
        old_days, new_days = 18+i, 5+i
        owner = ["planning desk", "asset records team", "document controller", "maintenance coordinator", "records supervisor"][i % 5]
        base = {"family": family, "source_type": "synthetic_policy", "page": 1,
                "source_url": "synthetic://production-proxy-v2", "source_record": family,
                "asset": asset, "site": site}

        def add_doc(suffix, text, roles, start="0001-01-01", end="9999-12-31"):
            d = dict(base, id=f"{family}-{suffix}", title=f"{site} {asset} administrative records {suffix}",
                     text=text, roles=roles, valid_from=start, valid_to=end)
            documents.append(d)
            return d

        old_quote = f"The archival review interval is {old_days} calendar days."
        new_quote = f"The archival review interval is {new_days} calendar days."
        old = add_doc("archive-v1", f"SIMULATED DOCUMENT. {site} station, asset {asset}. Administrative archive procedure, revision 1. Effective 2023-01-01 through 2024-12-31. {old_quote} This document governs record review only, not equipment operation.", ["operator"], "2023-01-01", "2024-12-31")
        new = add_doc("archive-v2", f"SIMULATED DOCUMENT. {site} station, asset {asset}. Administrative archive procedure, revision 2. Effective 2025-01-01. Supersedes revision 1. {new_quote} An incomplete archive package goes to the {owner}. Review frequency here is an administrative record requirement.", ["operator"], "2025-01-01")
        table_quote = f"{asset} | archive package | {2+i} attachments | FORM-{71+i}"
        table = add_doc("register", f"SIMULATED REGISTER. {site} station. Asset | package type | required count | form\n{table_quote}\nPX-{410+i}B | archive package | {9+i} attachments | FORM-{91+i}\nThe suffix A and B identify different assets; do not transfer a row between assets.", ["operator"])
        finance_quote = f"The internal budget reference is BUD-{830+i} and the review owner is finance cell {i+1}."
        add_doc("finance", f"SIMULATED FINANCE NOTE. {site} station {asset}. {finance_quote}", ["finance"])
        # Same equipment identifier in another site: requires site clarification.
        other = add_doc("other-site", f"SIMULATED DOCUMENT. Remote-{i} station, asset {asset}. The archival review interval is {31+i} calendar days. This is a different station with a locally reused identifier.", ["operator"], "2025-01-01")
        other["site"] = f"Remote-{i}"
        other["title"] = f"Remote-{i} {asset} administrative records other-site"

        def ev(d, quote):
            s = d["text"].index(quote)
            return [{"document_id": d["id"], "start": s, "end": s+len(quote)}]

        items = [
            ("current_version", f"At {site}, what is the current archival review interval for {asset}?", "retrieve", [ev(new, new_quote)], "2026-01-01", site),
            ("historical_version", f"As of June 2024, how often did {site} review the archive for {asset}?", "retrieve", [ev(old, old_quote)], "2024-06-01", site),
            ("identifier", f"For {site} {asset}, who receives an incomplete archive package?", "retrieve", [ev(new, f"An incomplete archive package goes to the {owner}.")], "2026-01-01", site),
            ("table", f"For {site} {asset}, how many attachments and which form does the archive register require?", "retrieve", [ev(table, table_quote)], "2026-01-01", site),
            ("multi_evidence", f"For {site} {asset}, give both today's review interval and the required attachment count and form.", "retrieve", [ev(new, new_quote), ev(table, table_quote)], "2026-01-01", site),
            ("permission_denied", f"Give the internal budget reference and finance review owner for {site} {asset}.", "no_authorized_evidence", [], "2026-01-01", site),
            ("unanswerable", f"What was the measured compressor bearing vibration for {asset} at {site} at 03:17 on 2025-09-09?", "insufficient_evidence", [], "2026-01-01", site),
            ("clarification", f"How frequently should the records for {asset} be reviewed now?", "clarify_site", [], "2026-01-01", None),
        ]
        for category, query, action, evidence, date, user_site in items:
            questions.append({"id": f"{family}-{category}", "family": family, "split": split,
                              "query": query, "category": category, "source_type": "synthetic_policy",
                              "expected_action": action, "evidence_groups": evidence,
                              "context": {"roles": ["operator"], "as_of": date, "site": user_site, "asset": asset},
                              "label_status": "synthetic fixtures; not observed customer requests"})
    write_jsonl(DATA / "documents.jsonl", documents)
    write_jsonl(DATA / "queries.jsonl", questions)
    (DATA / "source_audit.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")
    frozen_paths = [DATA / "documents.jsonl", DATA / "queries.jsonl", ROOT / "facts.tsv",
                    ROOT / "selection.json", ROOT / "PROTOCOL.md", ROOT / "build_dataset.py",
                    ROOT.parents[1] / "backend/Embeddings.py"]
    hashes = {str(p.relative_to(ROOT.parents[1])): sha(p) for p in frozen_paths}
    (DATA / "frozen_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps({"pages_or_records": len(documents), "queries": len(questions),
                      "public_questions": sum(q["source_type"] == "public_ocr" for q in questions),
                      "test_questions": sum(q["split"] == "test" for q in questions),
                      "reports_with_export_gaps": sum(bool(a["missing_between_first_last"]) for a in audits)}, indent=2))


if __name__ == "__main__":
    main()
