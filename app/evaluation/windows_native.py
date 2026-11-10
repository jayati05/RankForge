"""Read-only Win64 counters/virtual-region census; no NumPy/PyTorch imports."""

import ctypes as c
import os
from collections import Counter
from ctypes import wintypes as w


def libraries():
    if os.name != "nt" or c.sizeof(c.c_void_p) != 8:
        raise RuntimeError("Native region inspection requires Windows x64")
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    return kernel


def threads(pid: int) -> int:
    kernel = libraries()

    class Entry(c.Structure):
        _fields_ = [
            ("size", w.DWORD),
            ("usage", w.DWORD),
            ("tid", w.DWORD),
            ("pid", w.DWORD),
            ("priority", w.LONG),
            ("delta", w.LONG),
            ("flags", w.DWORD),
        ]

    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.Thread32First.argtypes = [w.HANDLE, c.POINTER(Entry)]
    kernel.Thread32Next.argtypes = [w.HANDLE, c.POINTER(Entry)]
    handle = kernel.CreateToolhelp32Snapshot(4, 0)
    if handle == w.HANDLE(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        entry = Entry()
        entry.size = c.sizeof(entry)
        count = 0
        valid = kernel.Thread32First(handle, c.byref(entry))
        while valid:
            if entry.pid == pid:
                count += 1
            valid = kernel.Thread32Next(handle, c.byref(entry))
        return count
    finally:
        kernel.CloseHandle(handle)


def counters(pid: int) -> dict:
    kernel = libraries()
    psapi = c.WinDLL("psapi", use_last_error=True)

    class Counters(c.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [
            (name, c.c_size_t)
            for name in (
                "peak_ws",
                "ws",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    psapi.GetProcessMemoryInfo.argtypes = [w.HANDLE, c.POINTER(Counters), w.DWORD]
    handle = kernel.OpenProcess(0x0410, False, pid)
    if not handle:
        raise c.WinError(c.get_last_error())
    try:
        value = Counters()
        value.cb = c.sizeof(value)
        if not psapi.GetProcessMemoryInfo(handle, c.byref(value), value.cb):
            raise c.WinError(c.get_last_error())
        return {
            "rss_bytes": value.ws,
            "os_peak_rss_bytes": value.peak_ws,
            "private_commit_bytes": value.private,
            "native_threads": threads(pid),
        }
    finally:
        kernel.CloseHandle(handle)


def regions(pid: int, details: bool = True) -> dict:
    kernel = libraries()
    psapi = c.WinDLL("psapi", use_last_error=True)

    class Info(c.Structure):
        _fields_ = [
            ("base", c.c_uint64),
            ("allocation_base", c.c_uint64),
            ("allocation_protect", w.DWORD),
            ("alignment", w.DWORD),
            ("size", c.c_uint64),
            ("state", w.DWORD),
            ("protect", w.DWORD),
            ("kind", w.DWORD),
            ("alignment2", w.DWORD),
        ]

    kernel.VirtualQueryEx.argtypes = [w.HANDLE, c.c_void_p, c.POINTER(Info), c.c_size_t]
    kernel.VirtualQueryEx.restype = c.c_size_t
    psapi.GetMappedFileNameW.argtypes = [w.HANDLE, c.c_void_p, w.LPWSTR, w.DWORD]
    handle = kernel.OpenProcess(0x0410, False, pid)
    if not handle:
        raise c.WinError(c.get_last_error())
    totals = Counter()
    image_files = Counter()
    largest = []
    address = 0
    count = 0
    try:
        while count < 200000:
            info = Info()
            returned = kernel.VirtualQueryEx(
                handle, c.c_void_p(address), c.byref(info), c.sizeof(info)
            )
            if not returned:
                break
            if info.size == 0 or info.base + info.size <= address:
                raise RuntimeError("Invalid native memory-region traversal")
            address = info.base + info.size
            count += 1
            if info.state not in (0x1000, 0x2000):
                continue
            state = "committed" if info.state == 0x1000 else "reserved"
            kind = {0x20000: "private", 0x40000: "mapped", 0x1000000: "image"}.get(
                info.kind, "other"
            )
            totals[state + "_bytes"] += info.size
            totals[state + "_" + kind + "_bytes"] += info.size
            if details and info.state == 0x1000:
                largest.append(
                    {
                        "base": hex(info.base),
                        "bytes": info.size,
                        "kind": kind,
                        "protection": info.protect,
                    }
                )
            if details and info.state == 0x1000 and kind in ("image", "mapped"):
                name = c.create_unicode_buffer(32768)
                if psapi.GetMappedFileNameW(handle, c.c_void_p(info.base), name, len(name)):
                    image_files[name.value] += info.size
        if count >= 200000:
            raise RuntimeError("Native memory-region safety bound exceeded")
        totals["virtual_allocated_bytes"] = totals["committed_bytes"] + totals["reserved_bytes"]
        return {
            "pid": pid,
            "region_count": count,
            "totals": dict(totals),
            "largest_committed_regions": sorted(largest, key=lambda r: -r["bytes"])[:25],
            "mapped_files": [
                {"path": name, "committed_region_bytes": size}
                for name, size in image_files.most_common()
            ],
            "limitations": "VirtualQueryEx region kinds describe mappings, not allocation stacks/owners. Copy-on-write image/mapped pages retain their region type. Reserved address space is not committed or resident memory.",
        }
    finally:
        kernel.CloseHandle(handle)
