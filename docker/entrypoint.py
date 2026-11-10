"""Container-only required-file preflight, then replace PID 1 with the API."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402


def required_files(settings):
    return (
        settings.dataset_dir / "corpus.jsonl",
        settings.dense_model_dir / "rankforge_model.json",
        settings.rerank_model_dir / "rankforge_model.json",
        settings.dense_index,
        settings.policy_dir / "protocol.json",
        settings.policy_dir / "frozen_policy.json",
    )


def main():
    settings = Settings.from_env()
    if any(not p.is_file() for p in required_files(settings)):
        raise SystemExit(
            "Required container resources missing; prepare and mount dataset/models/index/policy."
        )
    os.execv(sys.executable, [sys.executable, "-m", "app.api"])


if __name__ == "__main__":
    main()
