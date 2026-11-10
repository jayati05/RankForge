"""Download the pinned BEIR SciFact archive without adding BEIR dependencies."""

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.data.loaders import load_dataset

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_EXPANDED_BYTES = 250 * 1024 * 1024


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    if file_hash(path, "md5") != SCIFACT_MD5:
        raise ValueError("SciFact archive MD5 mismatch; refusing changed or corrupt archive")


def import_archive(archive: Path, destination: Path) -> Path:
    """Validate pinned bytes, copy only known members, then atomically publish dataset."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing overwrite: {destination}")
    verify_archive(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scifact-stage-", dir=destination.parent) as staging:
        root = Path(staging) / "scifact"
        root.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            for relative in MEMBERS:
                name = "scifact/" + relative
                if names.count(name) != 1:
                    raise ValueError(f"Archive requires exactly one {name}")
            if sum(bundle.getinfo("scifact/" + p).file_size for p in MEMBERS) > MAX_EXPANDED_BYTES:
                raise ValueError("Expanded dataset exceeds size limit")
            # No extractall: archive paths never determine a filesystem destination.
            for relative in MEMBERS:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open("scifact/" + relative) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        counts = {}
        for split in ("train", "test"):
            data = load_dataset(root, split)
            counts[split] = {
                "queries": len(data.queries),
                "judgments": sum(len(row) for row in data.qrels.values()),
            }
        metadata = {
            "name": "beir-scifact",
            "synthetic": False,
            "version": "BEIR archive MD5:" + SCIFACT_MD5,
            "source_url": SCIFACT_URL,
            "source_archive_sha256": file_hash(archive),
            "source_archive_md5": SCIFACT_MD5,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "license": "CC-BY-SA-4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "license_source": "https://huggingface.co/datasets/BeIR/scifact",
            "checksum_source": "https://github.com/beir-cellar/beir#beers-available-datasets",
            "format_source": "https://github.com/beir-cellar/beir/blob/main/beir/datasets/data_loader.py",
            "attribution": "SciFact: Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020; BEIR: Thakur et al., 2021.",
            "original_project": "https://github.com/allenai/scifact",
            "changes": "Selected four BEIR data files copied byte-for-byte; no text, IDs or judgments modified. Added this provenance metadata.",
            "document_count": len(data.documents),
            "splits": counts,
            "files_sha256": {p: file_hash(root / p) for p in MEMBERS},
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # rename, rather than replace: an existing destination is never intentionally overwritten.
        root.rename(destination)
    return destination


def download_scifact(destination: Path, archive: Path | None = None) -> Path:
    destination = Path(destination)
    if destination.exists():
        # Idempotent reuse only if provenance and every data file still match.
        metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("source_archive_md5") != SCIFACT_MD5:
            raise ValueError("Existing destination is not the pinned SciFact dataset")
        hashes = metadata.get("files_sha256", {})
        if set(hashes) != set(MEMBERS) or any(
            file_hash(destination / p) != hashes[p] for p in MEMBERS
        ):
            raise ValueError("Existing dataset differs from its provenance; use a new destination")
        for split in ("train", "test"):
            load_dataset(destination, split)
        return destination
    if archive is not None:
        return import_archive(Path(archive), destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scifact-download-", dir=destination.parent) as staging:
        downloaded = Path(staging) / "scifact.zip"
        request = urllib.request.Request(SCIFACT_URL, headers={"User-Agent": "RankForge/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, downloaded.open("wb") as sink:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Download exceeds size limit")
                sink.write(chunk)
        return import_archive(downloaded, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned BEIR SciFact (CC-BY-SA-4.0)")
    parser.add_argument("--destination", type=Path, default=Path("data/public/scifact"))
    parser.add_argument(
        "--archive", type=Path, help="Import an existing official scifact.zip offline"
    )
    args = parser.parse_args()
    try:
        print(download_scifact(args.destination, args.archive))
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        parser.exit(2, f"SciFact preparation failed: {exc}\nRetry: python -m app.data.download\n")


if __name__ == "__main__":
    main()
