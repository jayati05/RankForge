import csv
import json
import re
from pathlib import Path

from app.data.schemas import Dataset, Document, Query


def read_jsonl(path: Path, factory: type) -> dict:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL records must be objects")
                record = (
                    Document(row["_id"], row.get("title", ""), row["text"])
                    if factory is Document
                    else Query(row["_id"], row["text"])
                )
                if record.id in records:
                    raise ValueError(f"Duplicate ID: {record.id}")
                records[record.id] = record
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Empty dataset file: {path}")
    return records


def load_dataset(root: str | Path, split: str = "test") -> Dataset:
    """Load BEIR JSONL corpus/queries and TSV qrels, rejecting broken joins."""
    root = Path(root)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", split):
        raise ValueError("Split must be a simple name")
    documents = read_jsonl(root / "corpus.jsonl", Document)
    all_queries = read_jsonl(root / "queries.jsonl", Query)
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Expected qrels header: query-id, corpus-id, score")
        for line_number, row in enumerate(reader, 2):
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Qrel must contain exactly three TSV fields")
                qid, did = row["query-id"], row["corpus-id"]
                grade = int(row["score"])
                if grade < 0 or grade > 100:
                    raise ValueError("Grades must be integers between 0 and 100")
                if qid not in all_queries or did not in documents:
                    raise ValueError("Qrel references unknown query or document")
                if did in qrels.setdefault(qid, {}):
                    raise ValueError("Duplicate query-document judgment")
                qrels[qid][did] = grade
            except (ValueError, TypeError) as exc:
                raise ValueError(f"qrels line {line_number}: {exc}") from exc
    if not qrels:
        raise ValueError("Split has no judgments")
    queries = {qid: all_queries[qid] for qid in sorted(qrels)}
    metadata_path = root / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    if not isinstance(metadata, dict):
        raise ValueError("Dataset metadata must be a JSON object")
    return Dataset(documents, queries, qrels, metadata)
