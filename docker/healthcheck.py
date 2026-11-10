"""Cheap existing health endpoint; never initialize a model for health checks."""

import json
import os
from urllib.request import ProxyHandler, build_opener


def main():
    port = int(os.environ.get("RANKFORGE_PORT", "8000"))
    with build_opener(ProxyHandler({})).open(
        f"http://127.0.0.1:{port}/health", timeout=3
    ) as response:
        if json.load(response)["status"] != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
