"""Benchmark-only diagnostics. No HTTP controls, forced collection or cache eviction."""

import asyncio
import ctypes
import gc
import json
import os
import threading
import tracemalloc
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from app.evaluation.api_load import save
from app.evaluation.api_workload import METHODS


def read_control(path: Path) -> dict:
    """Atomic replacement can briefly deny opens on Windows; caller retries safely."""
    return json.loads(path.read_text(encoding="utf-8"))


def system_memory() -> dict:
    if os.name == "nt":
        from ctypes import wintypes

        class Status(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total",
                    "available",
                    "page_total",
                    "page_available",
                    "virtual_total",
                    "virtual_available",
                    "extended",
                )
            ]

        value = Status()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise ctypes.WinError()
        return {"system_total_bytes": value.total, "system_available_bytes": value.available}
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return {
        "system_total_bytes": int(info["MemTotal"].split()[0]) * 1024,
        "system_available_bytes": int(info["MemAvailable"].split()[0]) * 1024,
    }


def object_counts() -> dict:
    objects = gc.get_objects()  # Observe tracked objects; do not trigger collection.
    # Native extension metaclasses can expose a descriptor instead of a module string.
    counts = Counter(
        (type(obj).__module__ if isinstance(type(obj).__module__, str) else "<native-descriptor>")
        + "."
        + type(obj).__name__
        for obj in objects
    )
    return {
        "tracked_objects": len(objects),
        "gc_generation_counts": list(gc.get_count()),
        "selected_types": {
            name: number
            for name, number in counts.items()
            if any(
                s in name.lower()
                for s in ("request", "response", "task", "tensor", "searchresult", "future")
            )
        },
    }


class Diagnostics:
    def __init__(self):
        self.baseline = None

    def snapshot(self, manager, loop) -> dict:
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            snapshot = tracemalloc.take_snapshot()
            if self.baseline is None:
                self.baseline = snapshot
            trace = {
                "current_bytes": current,
                "peak_bytes": peak,
                "tracer_overhead_bytes": tracemalloc.get_tracemalloc_memory(),
                "top_current": [
                    {"location": str(s.traceback), "bytes": s.size, "count": s.count}
                    for s in snapshot.statistics("lineno")[:25]
                ],
                "top_difference_from_first_snapshot": [
                    {
                        "location": str(s.traceback),
                        "bytes_difference": s.size_diff,
                        "count_difference": s.count_diff,
                    }
                    for s in snapshot.compare_to(self.baseline, "lineno")[:25]
                ],
            }
        else:
            trace = None
        models = {}
        for name in ("encoder", "scorer"):
            adapter = manager.get(name)
            module = adapter.model if name == "encoder" else adapter.model.model
            models[name] = {
                "parameter_bytes": sum(p.numel() * p.element_size() for p in module.parameters()),
                "buffer_bytes": sum(p.numel() * p.element_size() for p in module.buffers()),
            }

        async def tasks():
            return dict(Counter(t.get_coro().__qualname__ for t in asyncio.all_tasks()))

        return {
            "pid": os.getpid(),
            "tracemalloc": trace,
            "objects": object_counts(),
            "python_threads": threading.active_count(),
            "async_tasks": asyncio.run_coroutine_threadsafe(tasks(), loop).result(timeout=10),
            "models": models,
            "dense_index_bytes": manager.get("dense").vectors.nbytes,
            "resources": manager.snapshot(),
            "notes": "Tracked-object counts exclude untracked/native allocations. Logical tensor bytes are not total resident model memory. One baseline tracemalloc Snapshot is intentionally retained; diagnostic temporary lists are released normally. No forced GC.",
        }


def serve(port: int, control_file: Path) -> None:
    import uvicorn
    from app.api.main import create_app

    app = create_app()
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application):
            application.state.diagnostic_loop = asyncio.get_running_loop()
            yield

    app.router.lifespan_context = lifespan
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False))
    diagnostics = Diagnostics()

    def control():
        previous = -1
        transient_reads = 0
        while not server.should_exit:
            try:
                message = read_control(control_file)
            except (OSError, ValueError):
                transient_reads += 1
                threading.Event().wait(0.05)
                continue
            if message["sequence"] <= previous:
                threading.Event().wait(0.05)
                continue
            previous = message["sequence"]
            command = message["command"]
            reply = {
                "command": command,
                "sequence": previous,
                "status": "complete",
                "transient_control_reads": transient_reads,
            }
            try:
                if command == "stop":
                    server.should_exit = True
                elif command == "idle":
                    pass
                else:
                    manager = app.state.resources
                    with manager.execution():
                        if command == "initialize":
                            for name in METHODS:
                                manager.get(name)
                        elif command == "trace_start":
                            tracemalloc.start(1)
                        elif command == "snapshot":
                            reply["diagnostics"] = diagnostics.snapshot(
                                manager, app.state.diagnostic_loop
                            )
                        elif command != "idle":
                            raise ValueError("Unknown benchmark control command")
            except Exception as exc:
                reply.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            destination = control_file.parent / f"command_{previous}.json"
            temporary = destination.with_suffix(".tmp")
            save(temporary, reply)
            temporary.replace(destination)

    threading.Thread(target=control, daemon=True).start()
    server.run()
