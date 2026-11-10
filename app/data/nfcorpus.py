"""Official BEIR NFCorpus preparation, gated on the frozen transfer protocol."""

import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.data.download import MAX_ARCHIVE_BYTES, MAX_EXPANDED_BYTES, file_hash
from app.data.loaders import load_dataset, read_jsonl
from app.data.schemas import Query
from app.evaluation.transfer_policy import load_protocol

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
NFCORPUS_MD5 = "a89dba18a62ef92f7d323ec890a0d38d"
SPLITS = ("train", "dev", "test")
MEMBERS = ("corpus.jsonl", "queries.jsonl", *(f"qrels/{s}.tsv" for s in SPLITS))


def verify_archive(archive: Path) -> None:
    if file_hash(archive, "md5") != NFCORPUS_MD5:
        raise ValueError("NFCorpus archive MD5 mismatch; refusing changed/corrupt bytes")


def prepare_nfcorpus(archive: Path, destination: Path, protocol_path: Path) -> Path:
    protocol, _ = load_protocol(protocol_path)  # Must precede any external qrel parsing.
    archive, destination = Path(archive), Path(destination)
    verify_archive(archive)
    if destination.exists():
        metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("source_archive_md5") != NFCORPUS_MD5 or metadata.get("files_sha256") != {
            p: file_hash(destination / p) for p in MEMBERS
        }:
            raise ValueError("Existing NFCorpus destination changed; choose a new destination")
        if metadata["transfer_policy_record_sha256"] != protocol["policy_record_sha256"]:
            raise ValueError("Existing data was prepared under a different transfer policy")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nfcorpus-stage-", dir=destination.parent) as temp:
        root = Path(temp) / "nfcorpus"
        root.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            for p in MEMBERS:
                if names.count("nfcorpus/" + p) != 1:
                    raise ValueError(f"Archive requires exactly one nfcorpus/{p}")
            if sum(bundle.getinfo("nfcorpus/" + p).file_size for p in MEMBERS) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded dataset exceeds size limit")
            for p in MEMBERS:
                target = root / p
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open("nfcorpus/" + p) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        statistics = {}
        query_sets = {}
        for split in SPLITS:
            data = load_dataset(root, split)
            query_sets[split] = set(data.queries)
            grades = Counter(grade for row in data.qrels.values() for grade in row.values())
            statistics[split] = {
                "query_count": len(data.queries),
                "judgment_count": sum(grades.values()),
                "grade_counts": dict(sorted(grades.items())),
                "queries_with_no_positive_judgments": sum(
                    not any(g > 0 for g in row.values()) for row in data.qrels.values()
                ),
            }
        raw_queries = read_jsonl(root / "queries.jsonl", Query)
        metadata = {
            "name": "beir-nfcorpus",
            "synthetic": False,
            "version": "BEIR archive MD5:" + NFCORPUS_MD5,
            "source_url": NFCORPUS_URL,
            "source_archive_md5": NFCORPUS_MD5,
            "source_archive_sha256": file_hash(archive),
            "archive_path": str(archive.resolve()),
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "transfer_frozen_at_utc": protocol["created_at_utc"],
            "transfer_policy_record_sha256": protocol["policy_record_sha256"],
            "license": "CC-BY-SA-4.0",
            "license_source": "https://huggingface.co/datasets/BeIR/nfcorpus",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "checksum_source": "https://github.com/beir-cellar/beir#beers-available-datasets",
            "original_project": "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
            "attribution": "NFCorpus: Boteva, Gholipour, Sokolov and Riezler, A Full-Text Learning to Rank Dataset for Medical Information Retrieval, ECIR 2016; BEIR: Thakur et al., 2021.",
            "document_count": len(data.documents),
            "raw_query_count": len(raw_queries),
            "splits": statistics,
            "split_query_id_overlaps": {
                f"{a}/{b}": len(query_sets[a] & query_sets[b])
                for i, a in enumerate(SPLITS)
                for b in SPLITS[i + 1 :]
            },
            "raw_query_ids_without_any_qrel": sorted(
                set(raw_queries) - set.union(*query_sets.values())
            ),
            "changes": "Required BEIR data files copied byte-for-byte; raw archive preserved. No documents, IDs, text or grades filtered or rewritten. Added provenance metadata.",
            "evaluation_split_selection": "Official test preselected in frozen protocol; train/dev inspected for format/counts only, not tuning",
            "unjudged_convention": "Zero evaluation gain, not known irrelevant",
            "files_sha256": {p: file_hash(root / p) for p in MEMBERS},
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        root.rename(destination)
    return destination


def download_nfcorpus(archive: Path, destination: Path, protocol_path: Path) -> Path:
    load_protocol(protocol_path)
    archive = Path(archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        import truststore

        truststore.inject_into_ssl()  # System trust, HTTPS verification enabled.
        with tempfile.TemporaryDirectory(prefix="nfcorpus-download-", dir=archive.parent) as temp:
            partial = Path(temp) / "nfcorpus.zip"
            request = urllib.request.Request(NFCORPUS_URL, headers={"User-Agent": "RankForge/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as sink,
            ):
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Download exceeds size limit")
                    sink.write(chunk)
            verify_archive(partial)
            partial.rename(archive)
    return prepare_nfcorpus(archive, destination, protocol_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official NFCorpus after policy freeze")
    parser.add_argument("--archive", type=Path, default=Path("data/public/archives/nfcorpus.zip"))
    parser.add_argument("--destination", type=Path, default=Path("data/public/nfcorpus"))
    parser.add_argument(
        "--protocol", type=Path, default=Path("experiments/configs/phase4-transfer")
    )
    args = parser.parse_args()
    try:
        print(download_nfcorpus(args.archive, args.destination, args.protocol))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(
            2,
            f"NFCorpus preparation failed: {exc}\nKeep the frozen protocol; retry with Windows Invoke-WebRequest and the same --archive path.\n",
        )


if __name__ == "__main__":
    main()
