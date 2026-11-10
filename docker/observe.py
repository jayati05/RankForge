"""Linux process RSS and cgroup-v2 counters; no model imports or HTTP endpoint."""

import json
from pathlib import Path
import time


def observe():
    result = {"unix_seconds": time.time()}
    root = Path("/sys/fs/cgroup")
    for name in ("memory.current", "memory.peak", "memory.max", "cpu.max"):
        path = root / name
        result[name] = path.read_text().strip() if path.is_file() else None
    for name in ("memory.events", "cpu.stat"):
        path = root / name
        result[name] = (
            dict(line.split() for line in path.read_text().splitlines()) if path.is_file() else None
        )
    status = dict(line.split(":", 1) for line in Path("/proc/1/status").read_text().splitlines())
    result["process_rss_bytes"] = int(status.get("VmRSS", "0 kB").split()[0]) * 1024
    result["process_peak_rss_bytes"] = int(status.get("VmHWM", "0 kB").split()[0]) * 1024
    result["process_threads"] = int(status["Threads"])
    return result


if __name__ == "__main__":
    print(json.dumps(observe()))
