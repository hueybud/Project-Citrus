#!/usr/bin/env python3
"""
CIT to CITF Batch Conversion Harness

Converts CIT replay files to CITF frame capture files by launching
Citrus Dolphin instances with null-backend/unlimited-speed playback.

Each CIT is a ZIP archive containing a .dtm input record, a .sav savestate,
and an output.json with match metadata. Dolphin replays the inputs and our
custom GameStateCapture code writes a .citframes file when the match ends.

Usage:
    python convert_cits.py <cit_folder>
    python convert_cits.py <cit_folder> --max-workers 2
    python convert_cits.py <cit_folder> --retry-failed
    python convert_cits.py <cit_folder> --dry-run
    python convert_cits.py <cit_folder> --no-enforce-comp-rules

Dependencies (pip install as needed):
    zstandard   -- for verifying CITF zstd magic (optional, graceful fallback)
"""

import argparse
import ctypes
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

if sys.platform == 'win32':
    import ctypes.wintypes as wintypes

try:
    import zstandard as _zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False

# analyze_citf lives in the same directory as this script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from analyze_citf import load_citf_bytes, parse_header, parse_frame
    _ANALYZE_AVAILABLE = True
except ImportError:
    _ANALYZE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — edit these paths if your layout differs
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == 'win32':
    DOLPHIN_EXE = r"C:\Users\Brian\source\repos\Project-Citrus2\Binary\x64\Citrus Dolphin.exe"
    ISO_PATH    = r"C:\Users\Brian\Downloads\Super Mario Strikers (USA)\Super Mario Strikers (USA).iso"
    DOLPHIN_INI = r"C:\Users\Brian\Documents\Dolphin Emulator\Config\Dolphin.ini"
else:
    DOLPHIN_EXE = os.environ.get('DOLPHIN_EXE', '/opt/dolphin/dolphin-emu-nogui')
    ISO_PATH    = os.environ.get('SMS_ISO',      '/data/game.iso')
    _cfg        = os.environ.get('XDG_CONFIG_HOME',
                                 os.path.join(os.path.expanduser('~'), '.config'))
    DOLPHIN_INI = os.environ.get('DOLPHIN_INI',
                                 os.path.join(_cfg, 'dolphin-emu', 'Dolphin.ini'))

MAX_CONCURRENT        = 3    # simultaneous Dolphin instances
STARTUP_WAIT_SECS     = 15   # flat wait before first memory check
STARTUP_TIMEOUT_SECS  = 90   # total seconds allowed for game to reach running state
MATCH_TIMEOUT_SECS    = 300  # floor for match timeout (used when time_elapsed is missing)
MATCH_TIMEOUT_BUFFER_SECS = 30   # buffer added on top of time_elapsed for dynamic timeout
POST_END_WAIT_SECS    = 10   # seconds after match-end flag before verifying CITF on disk
POLL_INTERVAL_SECS    = 2    # match-end polling frequency

# GameCube memory addresses (values stored big-endian in MEM1)
ADDR_MATCH_END    = 0x80400001  # u8:    1 = match ended and CITF has been written
ADDR_TIME_ELAPSED = 0x80400004  # float: elapsed game time in seconds (> 0 when running)

ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'   # zstd frame magic (little-endian on disk)
CITF_MIN_BYTES = 200                 # sanity-check minimum size

# Competitive rules enforced by --enforce-comp-rules (default: on).
# CITs that violate any of these are skipped and logged.
COMP_RULES: Dict = {
    "is_netplay":    True,
    "match_items":   True,
    "super_strikes": False,
    "bowser_ftx":    False,
    "time_allotted": "300",
    "difficulty":    "4",
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[str] = None, debug: bool = False) -> logging.Logger:
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dolphin.ini — enable null backend before any launches
# ──────────────────────────────────────────────────────────────────────────────

def set_null_backend(ini_path: str, enabled: bool) -> None:
    """
    Set [Movie] UseNullBackend = True/False in Dolphin.ini using a text-based approach
    that preserves all comments, blank lines, and section ordering.
    """
    target_str = "True" if enabled else "False"

    try:
        with open(ini_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        lines = []

    # Quick check: is it already set to the desired value?
    in_movie = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_movie = stripped.lower() == "[movie]"
        elif in_movie and "=" in stripped:
            key, _, val = stripped.partition("=")
            if key.strip().lower() == "usenullbackend":
                current = val.strip().lower() in ("true", "1", "yes")
                if current == enabled:
                    log.info("Dolphin.ini: [Movie] UseNullBackend already %s", target_str)
                    return
                break  # found but wrong value — will patch below

    # Patch or insert
    new_lines: List[str] = []
    in_movie = False
    found_section = False
    found_key = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            if in_movie and not found_key:
                # Left [Movie] section without seeing the key — insert before new section
                new_lines.append(f"UseNullBackend = {target_str}\n")
                found_key = True
            in_movie = stripped.lower() == "[movie]"
            if in_movie:
                found_section = True
        elif in_movie and not found_key and "=" in stripped:
            key = stripped.partition("=")[0].strip().lower()
            if key == "usenullbackend":
                line = f"UseNullBackend = {target_str}\n"
                found_key = True

        new_lines.append(line)

    # [Movie] was the last section and key wasn't seen
    if in_movie and not found_key:
        new_lines.append(f"UseNullBackend = {target_str}\n")
        found_key = True

    # [Movie] section doesn't exist at all
    if not found_section:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append("\n[Movie]\n")
        new_lines.append(f"UseNullBackend = {target_str}\n")

    Path(ini_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ini_path, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)
    log.info("Dolphin.ini: set [Movie] UseNullBackend = %s in %s", target_str, ini_path)


# ──────────────────────────────────────────────────────────────────────────────
# CIT metadata — read output.json from the ZIP archive
# ──────────────────────────────────────────────────────────────────────────────

def extract_cit_metadata(cit_path: Path) -> Dict:
    """
    Opens the CIT ZIP and parses output.json into a plain dict.
    Returns {} on any error so callers don't need to guard.
    Keys match the output.json field names used in ParseCITJson (Movie.cpp).
    """
    try:
        with zipfile.ZipFile(cit_path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not json_names:
                return {}
            with zf.open(json_names[0]) as jf:
                obj = json.loads(jf.read().decode("utf-8", errors="replace"))

        meta: Dict = {
            "epoch":          obj.get("Epoch", ""),
            "room_id":        obj.get("Room ID", ""),
            "game_count":     obj.get("Game Count", ""),
            "citrus_game_id": obj.get("Citrus Game Id", ""),
            "is_ranked":      obj.get("isRanked", "0") == "1",
            "is_netplay":     obj.get("Netplay Match", "0") == "1",
            "match_items":    obj.get("Match Items", "0") == "1",
            "super_strikes":  obj.get("Match Super Strikes", "0") == "1",
            "bowser_ftx":     obj.get("Match Bowser or FTX", "0") == "1",
            "time_allotted":  obj.get("Match Time Allotted", ""),
            "difficulty":     obj.get("Match Difficulty", ""),
            "time_elapsed":   obj.get("Match Time Elapsed", ""),
            "players":        [],
            "port_teams":     {},
        }

        # Player entries: [["P1 - Name", "discordId"], ...]
        meta["left_players"]  = []
        meta["right_players"] = []
        for entry in obj.get("Left Team Player Info", []):
            if isinstance(entry, list) and len(entry) >= 2:
                p = {"name": entry[0], "discord_id": entry[1]}
                meta["left_players"].append(p)
                meta["players"].append(p)
        for entry in obj.get("Right Team Player Info", []):
            if isinstance(entry, list) and len(entry) >= 2:
                p = {"name": entry[0], "discord_id": entry[1]}
                meta["right_players"].append(p)
                meta["players"].append(p)

        meta["port_teams"] = obj.get("Controller Port Info", {})

        # Goal timestamps: Goal Info[i][0] is the game-clock timestamp of the goal
        left_goals  = [entry[0] for entry in obj.get("Left Team Goal Info",  []) if isinstance(entry, list) and entry]
        right_goals = [entry[0] for entry in obj.get("Right Team Goal Info", []) if isinstance(entry, list) and entry]
        meta["json_goal_times"] = sorted(left_goals + right_goals)

        return meta

    except Exception as exc:
        log.warning("Could not extract metadata from %s: %s", cit_path.name, exc)
        return {}


def check_comp_rules(metadata: Dict) -> Optional[str]:
    """
    Check metadata against COMP_RULES.
    Returns a human-readable violation string if any rule fails, or None if all pass.
    """
    violations = []
    for field, expected in COMP_RULES.items():
        actual = metadata.get(field)
        if actual != expected:
            violations.append(f"{field}={actual!r} (expected {expected!r})")
    return (", ".join(violations)) if violations else None


def check_hvh_metadata(metadata: Dict) -> Optional[str]:
    """
    Check that exactly one human plays on each team.
    Returns a violation string if not HvH, or None if OK.
    A human is identified by a non-empty, non-zero discord_id.
    """
    def is_human(p: Dict) -> bool:
        did = str(p.get("discord_id", "")).strip()
        return bool(did) and did != "0"

    left_humans  = [p for p in metadata.get("left_players",  []) if is_human(p)]
    right_humans = [p for p in metadata.get("right_players", []) if is_human(p)]

    if len(left_humans) != 1:
        return f"left team has {len(left_humans)} human(s) (expected 1)"
    if len(right_humans) != 1:
        return f"right team has {len(right_humans)} human(s) (expected 1)"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Job tracker — JSON-backed persistent state
# ──────────────────────────────────────────────────────────────────────────────

class JobTracker:
    """
    Tracks conversion status for each CIT file across sessions.
    State is written atomically after every update.

    Statuses:
        running  — launched but not yet confirmed done (orphaned if process crashed)
        success  — CITF verified on disk
        failed   — gave up after timeout or hard error
        skipped  — intentionally excluded by --enforce-comp-rules
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._lock = Lock()
        self._state: Dict = {"jobs": {}}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as fh:
                    self._state = json.load(fh)
                jobs = self._state.get("jobs", {})
                log.info(
                    "Loaded state: %d jobs tracked (%d success / %d failed / %d skipped / %d running)",
                    len(jobs),
                    sum(1 for j in jobs.values() if j.get("status") == "success"),
                    sum(1 for j in jobs.values() if j.get("status") == "failed"),
                    sum(1 for j in jobs.values() if j.get("status") == "skipped"),
                    sum(1 for j in jobs.values() if j.get("status") == "running"),
                )
            except Exception as exc:
                log.warning("Could not load state file (%s) — starting fresh", exc)
                self._state = {"jobs": {}}

    def _save(self) -> None:
        try:
            tmp = str(self.state_file) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._state, fh, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as exc:
            log.error("Failed to save state file: %s", exc)

    def is_done(self, cit_name: str) -> bool:
        with self._lock:
            status = self._state["jobs"].get(cit_name, {}).get("status")
            return status in ("success", "failed", "skipped")

    def get_status(self, cit_name: str) -> Optional[str]:
        with self._lock:
            return self._state["jobs"].get(cit_name, {}).get("status")

    def mark_started(self, cit_name: str, metadata: Dict) -> None:
        with self._lock:
            self._state["jobs"][cit_name] = {
                "status":       "running",
                "attempted_at": _now_iso(),
                "metadata":     metadata,
            }
            self._save()

    def mark_success(self, cit_name: str, citf_path: str, citf_bytes: int) -> None:
        with self._lock:
            job = self._state["jobs"].setdefault(cit_name, {})
            job.update({
                "status":       "success",
                "completed_at": _now_iso(),
                "citf_path":    citf_path,
                "citf_bytes":   citf_bytes,
                "error":        None,
            })
            self._save()

    def mark_failed(self, cit_name: str, reason: str) -> None:
        with self._lock:
            job = self._state["jobs"].setdefault(cit_name, {})
            job.update({
                "status":       "failed",
                "completed_at": _now_iso(),
                "error":        reason,
            })
            self._save()

    def mark_skipped(self, cit_name: str, reason: str, metadata: Dict) -> None:
        with self._lock:
            self._state["jobs"][cit_name] = {
                "status":       "skipped",
                "completed_at": _now_iso(),
                "skip_reason":  reason,
                "metadata":     metadata,
            }
            self._save()

    def reset_job(self, cit_name: str) -> None:
        """Remove a job record so it will be retried."""
        with self._lock:
            self._state["jobs"].pop(cit_name, None)
            self._save()

    def summary(self) -> Dict:
        with self._lock:
            jobs = self._state["jobs"]
            return {
                "total":   len(jobs),
                "success": sum(1 for j in jobs.values() if j.get("status") == "success"),
                "failed":  sum(1 for j in jobs.values() if j.get("status") == "failed"),
                "skipped": sum(1 for j in jobs.values() if j.get("status") == "skipped"),
                "running": sum(1 for j in jobs.values() if j.get("status") == "running"),
            }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Dolphin process memory reader (cross-platform)
# ──────────────────────────────────────────────────────────────────────────────

if sys.platform == 'win32':
    class DolphinMemoryReader:
        """
        Reads emulated GameCube MEM1 from a live Dolphin process via ReadProcessMemory.

        Dolphin allocates MEM1 as an anonymous file-mapping (MEM_MAPPED) of 24 MB
        (0x1800000 bytes) for GameCube.  We locate it by scanning VirtualQueryEx for
        the largest committed PAGE_READWRITE mapped/private region >= 24 MB.

        GC address mapping:
            host_address = mem1_base + (gc_virt_addr - 0x80000000)

        GameCube memory is big-endian; floats must be read as big-endian.
        """

        PROCESS_VM_READ          = 0x0010
        PROCESS_QUERY_INFORMATION = 0x0400
        MEM_COMMIT   = 0x1000
        MEM_MAPPED   = 0x40000
        MEM_PRIVATE  = 0x20000
        PAGE_RW      = 0x04
        PAGE_RW_EXEC = 0x40

        GC_MEM1_SIZE = 0x1800000    # 24 MB
        GC_VIRT_BASE = 0x80000000   # GC kernel virtual base maps to physical 0

        class _MBI(ctypes.Structure):
            _fields_ = [
                ("BaseAddress",       ctypes.c_void_p),
                ("AllocationBase",    ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD),
                ("RegionSize",        ctypes.c_size_t),
                ("State",             wintypes.DWORD),
                ("Protect",           wintypes.DWORD),
                ("Type",              wintypes.DWORD),
            ]

        def __init__(self, pid: int):
            self._pid    = pid
            self._handle: Optional[int] = None
            self._base:   Optional[int] = None
            self._k32    = ctypes.windll.kernel32

        def open(self) -> bool:
            h = self._k32.OpenProcess(
                self.PROCESS_VM_READ | self.PROCESS_QUERY_INFORMATION, False, self._pid
            )
            if h:
                self._handle = h
            return bool(h)

        def close(self) -> None:
            if self._handle:
                self._k32.CloseHandle(self._handle)
                self._handle = None

        def find_mem1(self) -> bool:
            """
            Scan the process address space for the MEM1 region.

            Dolphin allocates GC RAM via CreateFileMapping + MapViewOfFileEx, so MEM1
            appears as MEM_COMMIT | PAGE_READWRITE | MEM_MAPPED (file-backed).

            We specifically exclude:
              - MEM_PRIVATE: covers JIT code cache, heaps, stacks
              - PAGE_EXECUTE_READWRITE (0x40): JIT code cache is 64 MB and would
                beat MEM1 (24 MB) in any size-based heuristic

            All matching candidates are collected and logged; the first (lowest host
            address) is used, since MEM1 is mapped at offset 0 of Dolphin's FastMem
            reservation and thus has the smallest base address.
            """
            if not self._handle:
                return False

            mbi        = self._MBI()
            addr       = 0
            candidates: List[Tuple[int, int]] = []   # (base, size)

            while addr < 0x7FFFFFFFFFFF:
                ret = self._k32.VirtualQueryEx(
                    self._handle, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)
                )
                if ret == 0:
                    break

                base = mbi.BaseAddress or 0
                size = mbi.RegionSize

                # Strict filter: PAGE_READWRITE (0x04) + MEM_MAPPED only.
                # This matches Dolphin's MapViewOfFileEx allocations (GC RAM mirrors)
                # and excludes the JIT code cache (PAGE_EXECUTE_READWRITE, MEM_PRIVATE).
                if (mbi.State == self.MEM_COMMIT
                        and mbi.Protect == self.PAGE_RW
                        and mbi.Type == self.MEM_MAPPED
                        and size >= self.GC_MEM1_SIZE):
                    candidates.append((base, size))

                addr = base + size
                if addr <= 0:
                    break

            if not candidates:
                return False

            log.debug("MEM1 scan: %d candidate(s): %s",
                      len(candidates),
                      ", ".join(f"base=0x{b:X} size=0x{s:X}" for b, s in candidates))

            # Use the first (lowest-address) candidate — MEM1 is at offset 0 of
            # Dolphin's FastMem reservation, so it has the smallest base address.
            self._base = candidates[0][0]
            return True

        @property
        def mem1_base(self) -> Optional[int]:
            return self._base

        def _host_addr(self, gc_virt: int) -> int:
            assert self._base is not None
            return self._base + (gc_virt - self.GC_VIRT_BASE)

        def _read_raw(self, gc_virt: int, n: int) -> Optional[bytes]:
            if not self._handle or self._base is None:
                return None
            buf       = ctypes.create_string_buffer(n)
            read_out  = ctypes.c_size_t(0)
            ok = self._k32.ReadProcessMemory(
                self._handle, ctypes.c_void_p(self._host_addr(gc_virt)),
                buf, n, ctypes.byref(read_out)
            )
            return bytes(buf.raw) if ok and read_out.value == n else None

        def read_u8(self, gc_virt: int) -> Optional[int]:
            data = self._read_raw(gc_virt, 1)
            return data[0] if data is not None else None

        def read_u32_be(self, gc_virt: int) -> Optional[int]:
            """Read a 4-byte big-endian unsigned int (GameCube native endian)."""
            data = self._read_raw(gc_virt, 4)
            return struct.unpack(">I", data)[0] if data is not None else None

        def read_f32_be(self, gc_virt: int) -> Optional[float]:
            """Read a 4-byte big-endian float (GameCube native endian)."""
            data = self._read_raw(gc_virt, 4)
            if data is None:
                return None
            try:
                val = struct.unpack(">f", data)[0]
                # Reject NaN / infinity which would indicate wrong region
                if not (val == val) or val != val or abs(val) > 1e9:
                    return None
                return val
            except struct.error:
                return None

else:
    class DolphinMemoryReader:
        """
        Reads emulated GameCube MEM1 from a live Dolphin process via /proc/{pid}/mem.

        Reads emulated GameCube MEM1 from a live Dolphin process.

        Two strategies, tried in order:
          1. Direct /dev/shm read: Dolphin creates /dev/shm/dolphin-emu.{pid};
             GC physical address 0 is at file offset 0, so reads use
             (gc_virt - 0x80000000) as the file offset.  Reliable on WSL2.
          2. /proc/{pid}/mem: scan /proc/maps for the MEM1 region, then read
             via process memory.  Works on bare-metal Linux and Docker.
        """

        GC_MEM1_SIZE = 0x1800000    # 24 MB
        GC_VIRT_BASE = 0x80000000

        # Pseudo-file entries that can never be MEM1.
        _SKIP_NAMES = frozenset(['[stack]', '[heap]', '[vdso]', '[vsyscall]', '[vvar]'])

        def __init__(self, pid: int):
            self._pid     = pid
            self._base:   Optional[int] = None
            self._fd:     Optional[int] = None   # /proc/pid/mem fd
            self._shm_fd: Optional[int] = None   # /dev/shm/dolphin-emu.{pid} fd

        def open(self) -> bool:
            # Prefer direct /dev/shm read — more reliable than /proc/pid/mem on
            # WSL2 where shared-memory regions return zeros via process_vm_readv.
            shm_path = f'/dev/shm/dolphin-emu.{self._pid}'
            try:
                self._shm_fd = os.open(shm_path, os.O_RDONLY)
                log.debug("Opened %s for direct shm read", shm_path)
                return True
            except OSError:
                pass
            # Fallback: /proc/pid/mem (works on bare-metal Linux / Docker)
            try:
                self._fd = os.open(f'/proc/{self._pid}/mem', os.O_RDONLY)
                return True
            except OSError as exc:
                log.debug("Could not open /proc/%d/mem: %s", self._pid, exc)
                return False

        def close(self) -> None:
            for attr in ('_shm_fd', '_fd'):
                fd = getattr(self, attr)
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    setattr(self, attr, None)

        def find_mem1(self) -> bool:
            """
            Locate GC MEM1.

            When Dolphin uses /dev/shm (WSL2 / older kernels), GC RAM lives at
            file offset 0 of /dev/shm/dolphin-emu.{pid}, so no virtual-address
            scan is needed — reads use (gc_virt - GC_VIRT_BASE) as the file offset.

            Otherwise scan /proc/{pid}/maps for the rw non-executable region that
            is exactly GC_MEM1_SIZE (24 MB) and backed by dolphin-emu shm or memfd.
            """
            if self._shm_fd is not None:
                # Direct shm mode: GC physical address 0 is at file offset 0.
                # Set _base = 0 so _host_addr() computes gc_virt - GC_VIRT_BASE.
                self._base = 0
                return True
            candidates: List[Tuple[int, int]] = []
            try:
                with open(f'/proc/{self._pid}/maps', 'r') as f:
                    for line in f:
                        parts = line.split(None, 5)
                        if len(parts) < 5:
                            continue
                        addr_range = parts[0]
                        perms      = parts[1]

                        # Must be readable + writable, NOT executable.
                        if len(perms) < 3 or perms[0] != 'r' or perms[1] != 'w' or perms[2] == 'x':
                            continue

                        # Skip known system pseudo-mappings.
                        name = parts[5].strip() if len(parts) > 5 else ''
                        if name in self._SKIP_NAMES:
                            continue

                        # Allow:
                        #   - Anonymous mappings (empty name)
                        #   - memfd-backed mappings (Dolphin FastMem, newer kernels)
                        #   - /dev/shm/dolphin-emu.* (Dolphin SharedMem, WSL2 / older kernels)
                        # Skip all other file-backed mappings (shared libraries, etc.)
                        if name.startswith('/') and 'memfd' not in name and 'dolphin-emu' not in name:
                            continue

                        start_str, end_str = addr_range.split('-')
                        start = int(start_str, 16)
                        end   = int(end_str,   16)
                        size  = end - start

                        if size >= self.GC_MEM1_SIZE:
                            candidates.append((start, size))

            except OSError as exc:
                log.debug("Failed to read /proc/%d/maps: %s", self._pid, exc)
                return False

            if not candidates:
                return False

            candidates.sort()  # ascending by base address
            log.debug("MEM1 scan: %d candidate(s): %s",
                      len(candidates),
                      ", ".join(f"base=0x{b:X} size=0x{s:X}" for b, s in candidates))
            # Prefer exact-size match (GC_MEM1_SIZE = 24MB) — on WSL2/Linux,
            # Dolphin maps several larger regions from the same shm file alongside
            # the one true 24MB MEM1 window.  Fall back to lowest-address otherwise.
            exact = [(b, s) for b, s in candidates if s == self.GC_MEM1_SIZE]
            self._base = (sorted(exact)[0] if exact else candidates[0])[0]
            return True

        @property
        def mem1_base(self) -> Optional[int]:
            return self._base

        def _host_addr(self, gc_virt: int) -> int:
            assert self._base is not None
            return self._base + (gc_virt - self.GC_VIRT_BASE)

        def _read_raw(self, gc_virt: int, n: int) -> Optional[bytes]:
            fd = self._shm_fd if self._shm_fd is not None else self._fd
            if fd is None or self._base is None:
                return None
            try:
                data = os.pread(fd, n, self._host_addr(gc_virt))
                return data if len(data) == n else None
            except OSError:
                return None

        def read_u8(self, gc_virt: int) -> Optional[int]:
            data = self._read_raw(gc_virt, 1)
            return data[0] if data is not None else None

        def read_u32_be(self, gc_virt: int) -> Optional[int]:
            """Read a 4-byte big-endian unsigned int (GameCube native endian)."""
            data = self._read_raw(gc_virt, 4)
            return struct.unpack(">I", data)[0] if data is not None else None

        def read_f32_be(self, gc_virt: int) -> Optional[float]:
            """Read a 4-byte big-endian float (GameCube native endian)."""
            data = self._read_raw(gc_virt, 4)
            if data is None:
                return None
            try:
                val = struct.unpack(">f", data)[0]
                # Reject NaN / infinity which would indicate wrong region
                if not (val == val) or val != val or abs(val) > 1e9:
                    return None
                return val
            except struct.error:
                return None


# ──────────────────────────────────────────────────────────────────────────────
# CITF validation
# ──────────────────────────────────────────────────────────────────────────────

def verify_citf(path: Path) -> Tuple[bool, str]:
    """
    Returns (ok, message).
    Checks: exists, minimum size, zstd magic bytes.
    """
    if not path.exists():
        return False, f"file not found: {path}"

    size = path.stat().st_size
    if size < CITF_MIN_BYTES:
        return False, f"suspiciously small ({size} bytes)"

    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
        if magic != ZSTD_MAGIC:
            # Warn but don't fail — older/uncompressed builds may differ
            return True, f"OK ({size:,} bytes, non-zstd magic {magic.hex()} — may be uncompressed)"
    except OSError as exc:
        return False, f"could not read file: {exc}"

    return True, f"OK ({size:,} bytes, zstd magic verified)"


def validate_citf_goals(citf_path: Path, json_goal_times: List[float],
                        tolerance: float = 0.1) -> Tuple[bool, str]:
    """
    Parse the CITF and derive goal timestamps from score-counter increments.
    Assert that the count and per-goal timestamps (sorted) match those recorded
    in output.json within `tolerance` seconds.

    Returns (ok, message).
    """
    if not _ANALYZE_AVAILABLE:
        return True, "analyze_citf not available — goal validation skipped"

    try:
        data = load_citf_bytes(str(citf_path))
        hdr  = parse_header(data)

        frames = []
        offset = hdr.header_size
        for _ in range(hdr.frame_count):
            frame, consumed = parse_frame(data, offset, hdr.fixed_frame_size)
            frames.append(frame)
            offset += consumed

        # Derive goals from score increments — no shooter attribution needed
        citf_goal_times = []
        for i in range(1, len(frames)):
            if frames[i].left_score > frames[i - 1].left_score:
                citf_goal_times.append(frames[i].game_time)
            if frames[i].right_score > frames[i - 1].right_score:
                citf_goal_times.append(frames[i].game_time)
        citf_goal_times.sort()

    except Exception as exc:
        return False, f"CITF parse error during goal validation: {exc}"

    n_citf = len(citf_goal_times)
    n_json = len(json_goal_times)

    if n_citf != n_json:
        return False, (
            f"goal count mismatch: CITF has {n_citf}, output.json has {n_json} "
            f"(CITF={citf_goal_times}, JSON={json_goal_times})"
        )

    mismatches = []
    for i, (ct, jt) in enumerate(zip(citf_goal_times, json_goal_times)):
        diff = abs(ct - jt)
        if diff > tolerance:
            mismatches.append(f"goal {i+1}: CITF={ct:.3f}s JSON={jt:.3f}s diff={diff:.3f}s")

    if mismatches:
        return False, "goal timestamp mismatch(es): " + "; ".join(mismatches)

    return True, f"{n_citf} goal(s) validated (timestamps within {tolerance}s)"


# ──────────────────────────────────────────────────────────────────────────────
# Single CIT conversion worker
# ──────────────────────────────────────────────────────────────────────────────

def convert_one_cit(
    cit_path: Path,
    tracker: JobTracker,
    startup_wait: float,
    startup_timeout: float,
    match_timeout: float,
    post_end_wait: float,
    dry_run: bool = False,
    enforce_comp_rules: bool = True,
) -> Optional[bool]:
    """
    Convert one CIT → CITF.

    Returns:
        True   — CITF produced and verified
        False  — conversion attempted but failed
        None   — skipped due to comp-rules enforcement (no Dolphin launched)

    Stages the CIT to an isolated temp directory before launching Dolphin so
    that simultaneous instances don't clobber each other's extracted files
    (all CIT ZIPs use the same internal names: output.dtm, output.dtm.sav, etc.).

    Match timeout is computed dynamically: max(match_timeout, time_elapsed + buffer).
    This ensures overtime matches (and other long games) are not cut off prematurely.
    """
    cit_name = cit_path.name
    stem     = cit_path.stem
    expected_citf = cit_path.parent / f"{stem}.citframes"

    # ── Extract metadata (no Dolphin needed) ─────────────────────────────────
    log.info("[%s] Extracting CIT metadata...", stem)
    metadata = extract_cit_metadata(cit_path)
    if metadata.get("players"):
        names = [p["name"] for p in metadata["players"]]
        log.info("[%s] Players: %s", stem, ", ".join(names))

    # ── Competitive rules check ───────────────────────────────────────────────
    if enforce_comp_rules:
        violation = check_comp_rules(metadata)
        if violation:
            log.info("[%s] SKIPPED (comp rules): %s", stem, violation)
            tracker.mark_skipped(cit_name, violation, metadata)
            return None

    # ── HvH check (always enforced) ──────────────────────────────────────────
    hvh_violation = check_hvh_metadata(metadata)
    if hvh_violation:
        log.info("[%s] SKIPPED (not HvH): %s", stem, hvh_violation)
        tracker.mark_skipped(cit_name, f"not HvH: {hvh_violation}", metadata)
        return None

    tracker.mark_started(cit_name, metadata)

    if dry_run:
        log.info("[%s] DRY RUN — skipping Dolphin launch", stem)
        tracker.mark_failed(cit_name, "dry_run")
        return False

    # ── Compute effective match timeout from time_elapsed + buffer ────────────
    effective_match_timeout = match_timeout
    time_elapsed_str = metadata.get("time_elapsed", "")
    if time_elapsed_str:
        try:
            game_time = float(time_elapsed_str)
            computed  = game_time + MATCH_TIMEOUT_BUFFER_SECS
            effective_match_timeout = max(match_timeout, computed)
            if effective_match_timeout > match_timeout:
                log.info(
                    "[%s] Match timeout: %.0fs (time_elapsed=%.1fs + %ds buffer > default %.0fs)",
                    stem, effective_match_timeout, game_time, MATCH_TIMEOUT_BUFFER_SECS, match_timeout,
                )
        except (ValueError, TypeError):
            log.warning("[%s] Could not parse time_elapsed %r — using default timeout %.0fs",
                        stem, time_elapsed_str, match_timeout)

    # ── Stage CIT to a per-job temp directory ────────────────────────────────
    job_tmp = Path(tempfile.mkdtemp(prefix=f"cit_{stem}_"))
    cit_copy = job_tmp / cit_name
    try:
        shutil.copy2(cit_path, cit_copy)
        log.info("[%s] Staged to %s", stem, job_tmp)
    except Exception as exc:
        shutil.rmtree(job_tmp, ignore_errors=True)
        reason = f"could not stage CIT: {exc}"
        log.error("[%s] %s", stem, reason)
        tracker.mark_failed(cit_name, reason)
        return False

    # ── Launch Dolphin ────────────────────────────────────────────────────────
    cmd = [DOLPHIN_EXE, "-m", str(cit_copy), "-e", ISO_PATH]
    if sys.platform != "win32":
        cmd += ["-p", "headless"]
    log.info("[%s] Launching: %s -m \"%s\" ...", stem, DOLPHIN_EXE, cit_name)

    dolphin_log = job_tmp / "dolphin.log"
    try:
        with open(dolphin_log, 'w') as dlf:
            proc = subprocess.Popen(cmd, stdout=dlf, stderr=dlf)
    except Exception as exc:
        shutil.rmtree(job_tmp, ignore_errors=True)
        reason = f"could not launch Dolphin: {exc}"
        log.error("[%s] %s", stem, reason)
        tracker.mark_failed(cit_name, reason)
        return False

    pid = proc.pid
    log.info("[%s] Dolphin PID %d — waiting %ds for startup...", stem, pid, int(startup_wait))
    time.sleep(startup_wait)

    # ── Check Dolphin is still alive ─────────────────────────────────────────
    if proc.poll() is not None:
        try:
            dolphin_output = dolphin_log.read_text(errors='replace').strip()
            if dolphin_output:
                log.error("[%s] Dolphin output:\n%s", stem, dolphin_output)
        except Exception:
            pass
        shutil.rmtree(job_tmp, ignore_errors=True)
        reason = f"Dolphin exited during startup (code {proc.returncode})"
        log.error("[%s] %s", stem, reason)
        tracker.mark_failed(cit_name, reason)
        return False

    # ── Open memory reader ────────────────────────────────────────────────────
    reader = DolphinMemoryReader(pid)
    mem_ok = False

    if reader.open() and reader.find_mem1():
        mem_ok = True
        log.info("[%s] MEM1 located at host 0x%X", stem, reader.mem1_base)
        # Sanity-check MEM1 by reading the left captain array pointer (0x8030d510).
        # A valid GC heap pointer is in the range 0x80000000–0x81FFFFFF.
        # A zero or obviously wrong value means we latched onto the wrong region.
        captain_ptr = reader.read_u32_be(0x8030d510)
        if captain_ptr is not None:
            looks_valid = 0x80000000 <= captain_ptr <= 0x81FFFFFF
            log.info("[%s] MEM1 sanity: 0x8030d510 = 0x%08X (%s)",
                     stem, captain_ptr, "looks valid" if looks_valid else "UNEXPECTED — may be wrong region")
            if not looks_valid:
                mem_ok = False
                log.warning("[%s] MEM1 sanity failed — falling back to file-based monitoring", stem)
        else:
            log.warning("[%s] MEM1 sanity: could not read 0x8030d510 — falling back to file-based monitoring", stem)
            mem_ok = False
    else:
        log.warning("[%s] MEM1 not found — falling back to file-based monitoring", stem)

    # ── Phase 1: confirm game is running (time elapsed > 0) ──────────────────
    startup_deadline = time.monotonic() + (startup_timeout - startup_wait)
    confirmed = not mem_ok  # if no memory access, skip directly to phase 2

    if mem_ok:
        log.info("[%s] Waiting for time elapsed > 0...", stem)
        while time.monotonic() < startup_deadline:
            if proc.poll() is not None:
                break
            t = reader.read_f32_be(ADDR_TIME_ELAPSED)
            if t is not None and t > 0.0:
                log.info("[%s] Game running (time=%.2fs)", stem, t)
                confirmed = True
                break
            time.sleep(POLL_INTERVAL_SECS)

        if not confirmed:
            if proc.poll() is None:
                # Process alive but memory check timed out — proceed optimistically
                log.warning("[%s] Could not confirm via memory after %ds — proceeding anyway",
                            stem, int(startup_timeout))
                confirmed = True
            else:
                _cleanup(proc, reader, job_tmp)
                reason = "Dolphin exited before game could be confirmed running"
                log.error("[%s] %s", stem, reason)
                tracker.mark_failed(cit_name, reason)
                return False

    # ── Phase 2: poll for match end ───────────────────────────────────────────
    log.info("[%s] Waiting for match end flag (0x%08X == 1), timeout=%.0fs...",
             stem, ADDR_MATCH_END, effective_match_timeout)
    match_deadline = time.monotonic() + effective_match_timeout
    match_ended    = False

    while time.monotonic() < match_deadline:
        if proc.poll() is not None:
            log.warning("[%s] Dolphin exited during match (code %s)", stem, proc.returncode)
            break

        if mem_ok:
            flag = reader.read_u8(ADDR_MATCH_END)
            if flag == 1:
                log.info("[%s] Match-end flag detected", stem)
                match_ended = True
                break
        else:
            # File-based fallback: CITF appears in temp dir once EndCapture() runs
            if (job_tmp / f"{stem}.citframes").exists():
                log.info("[%s] CITF file appeared in temp dir (file-monitor fallback)", stem)
                match_ended = True
                break

        time.sleep(POLL_INTERVAL_SECS)

    if not match_ended:
        timed_out = proc.poll() is None  # True = hit timeout; False = Dolphin crashed
        if timed_out:
            log.warning("[%s] Timeout after %.0fs — terminating Dolphin", stem, effective_match_timeout)
        _cleanup(proc, reader, job_tmp)
        reason = (
            f"match did not end within {int(effective_match_timeout)}s" if timed_out
            else f"Dolphin exited unexpectedly (code {proc.returncode})"
        )
        tracker.mark_failed(cit_name, reason)
        return False

    # ── Phase 3: wait for CITF flush, then copy out and kill Dolphin ─────────
    try:
        fps_matches = []
        for line in dolphin_log.read_text(errors='replace').splitlines():
            m = re.search(r'FPS:\s*(\d+)\s*-\s*VPS:\s*(\d+)\s*-\s*(\d+)%', line)
            if m:
                fps_matches.append(m)
                if len(fps_matches) == 10:
                    break
        if fps_matches:
            m = fps_matches[-1]
            log.info("[%s] FPS (stabilized): %s | VPS: %s | Speed: %s%%",
                     stem, m.group(1), m.group(2), m.group(3))
    except Exception:
        pass
    log.info("[%s] Waiting %ds for CITF to flush to disk...", stem, int(post_end_wait))
    time.sleep(post_end_wait)

    # Dolphin writes the CITF next to the staged CIT copy (in job_tmp).
    # Copy it to the original CIT's directory before we delete the temp dir.
    temp_citf = job_tmp / f"{stem}.citframes"
    if temp_citf.exists():
        # Validate goal count and timestamps against output.json before accepting the CITF
        json_goal_times = metadata.get("json_goal_times", [])
        goal_ok, goal_msg = validate_citf_goals(temp_citf, json_goal_times)
        if goal_ok:
            log.info("[%s] Goal validation: %s", stem, goal_msg)
        else:
            log.error("[%s] Goal validation FAILED: %s", stem, goal_msg)
            _cleanup(proc, reader, job_tmp)
            tracker.mark_failed(cit_name, f"goal validation failed: {goal_msg}")
            return False

        try:
            shutil.copy2(temp_citf, expected_citf)
            log.info("[%s] Copied CITF from temp dir to %s", stem, expected_citf)
        except Exception as exc:
            _cleanup(proc, reader, job_tmp)
            reason = f"could not copy CITF from temp dir: {exc}"
            log.error("[%s] %s", stem, reason)
            tracker.mark_failed(cit_name, reason)
            return False
    else:
        log.warning("[%s] CITF not found in temp dir (%s) — may have been written elsewhere",
                    stem, temp_citf)

    _cleanup(proc, reader, job_tmp)

    # ── Phase 4: verify CITF ──────────────────────────────────────────────────
    ok, msg = verify_citf(expected_citf)
    if not ok:
        reason = f"CITF verification failed: {msg}"
        log.error("[%s] %s", stem, reason)
        tracker.mark_failed(cit_name, reason)
        return False

    size = expected_citf.stat().st_size
    log.info("[%s] SUCCESS — %s — %s", stem, expected_citf.name, msg)
    tracker.mark_success(cit_name, str(expected_citf), size)
    return True


def _cleanup(proc: subprocess.Popen, reader: DolphinMemoryReader, job_tmp: Path) -> None:
    reader.close()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning("Dolphin PID %d did not exit on SIGTERM — killing", proc.pid)
            proc.kill()
            proc.wait()
    shutil.rmtree(job_tmp, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert CIT replay files to CITF frame captures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("cit_folder",          help="Directory containing .cit files")
    parser.add_argument("--max-workers",       type=int,   default=MAX_CONCURRENT,
                        help="Max simultaneous Dolphin instances")
    parser.add_argument("--startup-wait",      type=float, default=STARTUP_WAIT_SECS,
                        help="Flat seconds to wait before first memory check")
    parser.add_argument("--startup-timeout",   type=float, default=STARTUP_TIMEOUT_SECS,
                        help="Total seconds allowed for game startup confirmation")
    parser.add_argument("--match-timeout",     type=float, default=MATCH_TIMEOUT_SECS,
                        help="Floor for match timeout in real-time seconds. The actual timeout "
                             "is max(this, time_elapsed + %(default)ds buffer) when time_elapsed "
                             "is available in the CIT metadata.")
    parser.add_argument("--post-end-wait",     type=float, default=POST_END_WAIT_SECS,
                        help="Seconds after match-end flag before verifying CITF")
    parser.add_argument("--state-file",        default=None,
                        help="Path to job-state JSON (default: <cit_folder>/conversion_state.json)")
    parser.add_argument("--log-file",          default=None,
                        help="Also write logs to this file")
    parser.add_argument("--launch-stagger",    type=float, default=5.0,
                        help="Seconds to wait between launching each Dolphin instance "
                             "(prevents simultaneous startup dialogs)")
    parser.add_argument("--no-null-backend",   action="store_true",
                        help="Set UseNullBackend = False in Dolphin.ini "
                             "(runs at normal speed; useful for debugging)")
    parser.add_argument("--retry-failed",      action="store_true",
                        help="Re-attempt CITs previously marked as failed")
    parser.add_argument("--retry-skipped",     action="store_true",
                        help="Re-attempt CITs previously skipped by --enforce-comp-rules")
    parser.add_argument("--dry-run",           action="store_true",
                        help="Enumerate and log CITs without launching Dolphin")
    parser.add_argument("--debug",             action="store_true",
                        help="Enable DEBUG-level logging (shows MEM1 scan candidates etc.)")

    # Competitive rules enforcement — enabled by default; pass --no-enforce-comp-rules to disable.
    comp_group = parser.add_mutually_exclusive_group()
    comp_group.add_argument(
        "--enforce-comp-rules",
        dest="enforce_comp_rules", action="store_true", default=True,
        help="Skip CITs that don't meet competitive settings "
             "(is_netplay, items on, super_strikes off, bowser_ftx off, 300s, difficulty 4). "
             "Enabled by default.",
    )
    comp_group.add_argument(
        "--no-enforce-comp-rules",
        dest="enforce_comp_rules", action="store_false",
        help="Process all CITs regardless of competitive settings.",
    )

    args = parser.parse_args()

    setup_logging(args.log_file, debug=args.debug)

    if args.enforce_comp_rules:
        log.info("Comp-rules enforcement ON — skipping non-comp CITs "
                 "(pass --no-enforce-comp-rules to disable)")
    else:
        log.info("Comp-rules enforcement OFF — processing all CITs")

    # ── Validate paths ────────────────────────────────────────────────────────
    cit_folder = Path(args.cit_folder)
    if not cit_folder.is_dir():
        log.error("Not a directory: %s", cit_folder)
        sys.exit(1)

    if not Path(DOLPHIN_EXE).exists():
        log.error("Dolphin executable not found: %s", DOLPHIN_EXE)
        sys.exit(1)

    if not Path(ISO_PATH).exists():
        log.error("ISO not found: %s", ISO_PATH)
        sys.exit(1)

    # ── Configure Dolphin.ini before any launches ─────────────────────────────
    if not args.dry_run:
        set_null_backend(DOLPHIN_INI, not args.no_null_backend)

    # ── Discover CIT files ────────────────────────────────────────────────────
    cit_files = sorted(cit_folder.glob("*.cit"))
    if not cit_files:
        log.error("No .cit files found in %s", cit_folder)
        sys.exit(1)
    log.info("Found %d .cit files in %s", len(cit_files), cit_folder)

    # Warn about duplicate stems (they'd write to the same CITF path)
    stems: Dict[str, List[str]] = {}
    for c in cit_files:
        stems.setdefault(c.stem, []).append(c.name)
    for stem, names in stems.items():
        if len(names) > 1:
            log.warning("Duplicate stem '%s' — these CITs share the same output path: %s",
                        stem, names)

    # ── Load job tracker ──────────────────────────────────────────────────────
    state_path = Path(args.state_file) if args.state_file else (cit_folder / "conversion_state.json")
    tracker = JobTracker(state_path)

    # ── Build pending list ────────────────────────────────────────────────────
    pending: List[Path] = []
    skip_success = 0
    skip_failed  = 0
    skip_skipped = 0

    for cit in cit_files:
        status = tracker.get_status(cit.name)
        if status == "success":
            skip_success += 1
        elif status == "failed":
            if args.retry_failed:
                log.info("Retrying previously-failed: %s", cit.name)
                tracker.reset_job(cit.name)
                pending.append(cit)
            else:
                skip_failed += 1
        elif status == "skipped":
            if args.retry_skipped:
                log.info("Retrying previously-skipped: %s", cit.name)
                tracker.reset_job(cit.name)
                pending.append(cit)
            else:
                skip_skipped += 1
        elif status == "running":
            # Orphaned from a previous crashed session — retry
            log.info("Re-queuing orphaned-running job: %s", cit.name)
            tracker.reset_job(cit.name)
            pending.append(cit)
        else:
            pending.append(cit)

    skipped_hints = []
    if skip_failed and not args.retry_failed:
        skipped_hints.append("use --retry-failed to re-attempt failed")
    if skip_skipped and not args.retry_skipped:
        skipped_hints.append("use --retry-skipped to re-attempt skipped")

    log.info(
        "Skipped %d already-successful, %d previously-failed, %d previously-skipped%s | %d to process",
        skip_success, skip_failed, skip_skipped,
        (" (" + "; ".join(skipped_hints) + ")") if skipped_hints else "",
        len(pending),
    )

    if not pending:
        log.info("Nothing to do.")
        _print_summary(tracker)
        return

    # ── Process ───────────────────────────────────────────────────────────────
    log.info("Starting conversion: %d CITs | %d concurrent workers | %.0fs launch stagger",
             len(pending), args.max_workers, args.launch_stagger)

    success_n = 0
    failed_n  = 0
    skipped_n = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures: Dict = {}
        for i, cit in enumerate(pending):
            if i > 0 and args.launch_stagger > 0:
                log.info("Staggering next launch by %.0fs...", args.launch_stagger)
                time.sleep(args.launch_stagger)
            futures[pool.submit(
                convert_one_cit,
                cit, tracker,
                args.startup_wait,
                args.startup_timeout,
                args.match_timeout,
                args.post_end_wait,
                args.dry_run,
                args.enforce_comp_rules,
            )] = cit

        for future in as_completed(futures):
            cit = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = False
                log.exception("Unhandled exception processing %s: %s", cit.name, exc)
                tracker.mark_failed(cit.name, f"unhandled exception: {exc}")

            if result is True:
                success_n += 1
            elif result is None:
                skipped_n += 1
            else:
                failed_n += 1

            done = success_n + failed_n + skipped_n
            log.info("Progress: %d/%d  (%d ok, %d skipped, %d failed)",
                     done, len(pending), success_n, skipped_n, failed_n)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("COMPLETE — this run: %d success, %d skipped, %d failed out of %d processed",
             success_n, skipped_n, failed_n, len(pending))
    _print_summary(tracker)
    log.info("State file: %s", state_path)


def _print_summary(tracker: JobTracker) -> None:
    s = tracker.summary()
    log.info("Cumulative state: %d success / %d skipped / %d failed / %d total tracked",
             s["success"], s["skipped"], s["failed"], s["total"])


if __name__ == "__main__":
    main()
