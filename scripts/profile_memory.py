#!/usr/bin/env python3
"""Profile peak host memory used by a `retriever` invocation.

Spawns the given command (default: `retriever ingest ... --run-mode {batch,inprocess}`),
samples memory across the entire process tree plus a few system-wide signals at a
fixed cadence, and writes a summary + timeseries CSV.

What's tracked at every tick:
    - Per-process RSS / USS (sum across the whole tree spawned by the target)
    - /proc/meminfo MemAvailable delta vs. the pre-launch baseline
    - /dev/shm usage delta (Ray's plasma store lives here)
    - GPU memory (best-effort via nvidia-smi)

At the tick where total tree RSS peaks, a snapshot of the top-N processes is taken
so you can attribute the peak to specific actors/workers.

Usage:
    python scripts/profile_memory.py --mode batch  --documents '/home/edwardk/data/bo767/*.pdf'
    python scripts/profile_memory.py --mode inprocess --documents '/home/edwardk/data/bo767/1*.pdf'

The default command can be overridden completely with `-- <cmd> <args...>`:
    python scripts/profile_memory.py --label custom -- retriever ingest ... --run-mode batch
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import pty
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import BinaryIO, Optional

try:
    import psutil
except ImportError:
    print("psutil is required. pip install psutil", file=sys.stderr)
    sys.exit(2)


GIB = 1024**3


def _tee_stream(src: BinaryIO, *dests: BinaryIO) -> None:
    """Pump bytes from ``src`` to every destination until EOF.

    Used so the spawned target's stdout/stderr stream to the user's terminal
    *and* the captured log file simultaneously. ``src`` is typically a PTY
    master fd opened in binary mode — reading from a PTY raises ``OSError``
    (Errno 5) when the slave closes, which we treat as EOF.
    """
    try:
        while True:
            try:
                chunk = src.read(4096)
            except OSError:
                # PTY slave closed (child exited). Treat as EOF.
                break
            if not chunk:
                break
            for dest in dests:
                try:
                    dest.write(chunk)
                    dest.flush()
                except (BrokenPipeError, ValueError):
                    # ValueError: destination already closed (race during shutdown).
                    pass
    finally:
        try:
            src.close()
        except Exception:
            pass


def _open_pty_pair() -> tuple[int, int]:
    """Allocate a (master, slave) PTY pair and propagate the parent's terminal
    size so progress bars sized to the user's window render correctly."""
    master_fd, slave_fd = pty.openpty()
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        winsize = struct.pack("HHHH", size.lines, size.columns, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
    except (OSError, ValueError):
        # Non-tty parent or environment without ioctl; defaults are fine.
        pass
    return master_fd, slave_fd


def read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    # values reported in kB
                    out[key] = int(parts[1]) * 1024
                except ValueError:
                    pass
    return out


def shm_usage_bytes(path: str = "/dev/shm") -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                p = os.path.join(root, name)
                try:
                    st = os.lstat(p)
                    total += st.st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total


def gpu_memory_used_bytes() -> Optional[list[int]]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None
    try:
        out = subprocess.run(
            [nvsmi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    mibs: list[int] = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            mibs.append(int(s) * 1024 * 1024)
        except ValueError:
            continue
    return mibs


@dataclass
class ProcSample:
    pid: int
    cmdline: str
    rss: int
    uss: int


@dataclass
class Tick:
    t: float
    tree_rss: int
    tree_uss: int
    tree_count: int
    meminfo_used_vs_baseline: int  # baseline_avail - now_avail
    shm_bytes: int
    gpu_used: list[int] = field(default_factory=list)


def gather_tree(root: psutil.Process) -> tuple[int, int, int, list[ProcSample]]:
    """Return (sum_rss, sum_uss, count, samples). USS is best-effort."""
    procs: list[psutil.Process] = [root]
    try:
        procs.extend(root.children(recursive=True))
    except psutil.NoSuchProcess:
        pass

    sum_rss = 0
    sum_uss = 0
    count = 0
    samples: list[ProcSample] = []
    for p in procs:
        try:
            # memory_full_info() includes uss but requires permissions and is slower.
            mi_full = None
            try:
                mi_full = p.memory_full_info()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            mi = mi_full if mi_full is not None else p.memory_info()
            rss = getattr(mi, "rss", 0)
            uss = getattr(mi, "uss", 0)
            sum_rss += rss
            sum_uss += uss
            count += 1
            try:
                cmd = " ".join(p.cmdline()) or p.name()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                cmd = "<unknown>"
            samples.append(ProcSample(pid=p.pid, cmdline=cmd[:240], rss=rss, uss=uss))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return sum_rss, sum_uss, count, samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["batch", "inprocess"], default="batch",
                   help="Run mode for the default retriever ingest command.")
    p.add_argument("--documents", default="/home/edwardk/data/bo767/*.pdf",
                   help="Glob/path passed to retriever ingest in default command.")
    p.add_argument("--lancedb-uri", default=None,
                   help="LanceDB URI for retriever ingest. Defaults to ./lancedb_<mode>_<ts>.")
    p.add_argument("--table-name", default="profile_memory",
                   help="LanceDB table name for retriever ingest.")
    p.add_argument("--interval", type=float, default=0.5,
                   help="Sampling interval in seconds.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Top-N per-process snapshot to capture at peak.")
    p.add_argument("--out-dir", default="memprof_runs",
                   help="Directory to write summary + timeseries.")
    p.add_argument("--label", default=None,
                   help="Optional label for output filenames. Defaults to {mode}_{timestamp}.")
    p.add_argument("--timeout", type=float, default=None,
                   help="Kill the target after N seconds (whole tree). Useful for smoke tests.")
    p.add_argument("--no-shm", action="store_true",
                   help="Skip /dev/shm walk (cheaper, but loses plasma signal).")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="Optional explicit command after `--`. Overrides the default retriever ingest invocation.")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    label = args.label or f"{args.mode}_{ts}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{label}.summary.json"
    timeseries_path = out_dir / f"{label}.timeseries.csv"
    stdout_log = out_dir / f"{label}.stdout.log"
    stderr_log = out_dir / f"{label}.stderr.log"

    # Build command.
    explicit = [a for a in args.cmd if a != "--"]
    if explicit:
        cmd = explicit
    else:
        lancedb_uri = args.lancedb_uri or str(out_dir / f"lancedb_{label}")
        cmd = [
            "retriever", "ingest", args.documents,
            "--run-mode", args.mode,
            "--lancedb-uri", lancedb_uri,
            "--table-name", args.table_name,
        ]

    print(f"[profile] label={label}", file=sys.stderr)
    print(f"[profile] cmd={shlex.join(cmd)}", file=sys.stderr)
    print(f"[profile] out_dir={out_dir.resolve()}", file=sys.stderr)

    # Baselines.
    baseline_mem = read_meminfo()
    baseline_avail = baseline_mem.get("MemAvailable", 0)
    baseline_shm = 0 if args.no_shm else shm_usage_bytes()
    print(f"[profile] baseline MemAvailable = {baseline_avail / GIB:.2f} GiB; "
          f"baseline /dev/shm = {baseline_shm / GIB:.2f} GiB", file=sys.stderr)

    t0 = time.time()
    with open(stdout_log, "wb") as fo, open(stderr_log, "wb") as fe:
        # PTYs so the child sees its stdout/stderr as terminals — required for
        # libraries like Ray's rich-progress to keep updating the bar in place
        # via carriage returns instead of dumping a fresh line per tick.
        stdout_master, stdout_slave = _open_pty_pair()
        stderr_master, stderr_slave = _open_pty_pair()
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_slave,
            stderr=stderr_slave,
            preexec_fn=os.setsid,  # own process group so we can kill the tree
        )
        # Close the slave ends in the parent so reads on the masters can EOF
        # cleanly when the child exits.
        os.close(stdout_slave)
        os.close(stderr_slave)
        stdout_reader = os.fdopen(stdout_master, "rb", buffering=0)
        stderr_reader = os.fdopen(stderr_master, "rb", buffering=0)
        tee_threads = [
            threading.Thread(
                target=_tee_stream,
                args=(stdout_reader, sys.stdout.buffer, fo),
                daemon=True,
            ),
            threading.Thread(
                target=_tee_stream,
                args=(stderr_reader, sys.stderr.buffer, fe),
                daemon=True,
            ),
        ]
        for t in tee_threads:
            t.start()
        try:
            ps_root = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            print("[profile] target exited before we could attach", file=sys.stderr)
            return 1

        # Sampling loop.
        ticks: list[Tick] = []
        peak_rss = 0
        peak_samples: list[ProcSample] = []
        peak_tick: Optional[Tick] = None
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    # Capture a final tick after exit so we see teardown.
                    break

                if args.timeout is not None and (time.time() - t0) > args.timeout:
                    print(f"[profile] timeout {args.timeout}s reached — killing tree", file=sys.stderr)
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass

                try:
                    sum_rss, sum_uss, count, samples = gather_tree(ps_root)
                except psutil.NoSuchProcess:
                    break

                mi = read_meminfo()
                avail = mi.get("MemAvailable", baseline_avail)
                shm_b = 0 if args.no_shm else shm_usage_bytes()
                gpu = gpu_memory_used_bytes() or []

                tick = Tick(
                    t=time.time() - t0,
                    tree_rss=sum_rss,
                    tree_uss=sum_uss,
                    tree_count=count,
                    meminfo_used_vs_baseline=max(0, baseline_avail - avail),
                    shm_bytes=max(0, shm_b - baseline_shm),
                    gpu_used=gpu,
                )
                ticks.append(tick)

                if sum_rss > peak_rss:
                    peak_rss = sum_rss
                    peak_tick = tick
                    peak_samples = sorted(samples, key=lambda s: s.rss, reverse=True)[:args.top_n]

                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("[profile] interrupted — killing target", file=sys.stderr)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        ret = proc.wait()
        # Drain any trailing buffered output before closing the log files.
        for t in tee_threads:
            t.join(timeout=5.0)

    elapsed = time.time() - t0
    print(f"[profile] target exited rc={ret} after {elapsed:.1f}s; {len(ticks)} samples",
          file=sys.stderr)

    # Write timeseries CSV.
    with open(timeseries_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_seconds", "tree_rss_bytes", "tree_uss_bytes", "tree_count",
                    "meminfo_used_vs_baseline_bytes", "shm_delta_bytes", "gpu0_used_bytes",
                    "gpu1_used_bytes"])
        for tk in ticks:
            g0 = tk.gpu_used[0] if len(tk.gpu_used) > 0 else ""
            g1 = tk.gpu_used[1] if len(tk.gpu_used) > 1 else ""
            w.writerow([f"{tk.t:.3f}", tk.tree_rss, tk.tree_uss, tk.tree_count,
                        tk.meminfo_used_vs_baseline, tk.shm_bytes, g0, g1])

    # Aggregate peaks (across all ticks, not only the tree-rss-peak tick).
    peak_uss = max((t.tree_uss for t in ticks), default=0)
    peak_meminfo = max((t.meminfo_used_vs_baseline for t in ticks), default=0)
    peak_shm = max((t.shm_bytes for t in ticks), default=0)
    peak_count = max((t.tree_count for t in ticks), default=0)

    summary = {
        "label": label,
        "cmd": cmd,
        "returncode": ret,
        "elapsed_seconds": elapsed,
        "samples": len(ticks),
        "interval_seconds": args.interval,
        "baseline": {
            "mem_available_bytes": baseline_avail,
            "shm_bytes": baseline_shm,
        },
        "peak": {
            "tree_rss_bytes": peak_rss,
            "tree_rss_gib": peak_rss / GIB,
            "tree_uss_bytes": peak_uss,
            "tree_uss_gib": peak_uss / GIB,
            "meminfo_used_vs_baseline_bytes": peak_meminfo,
            "meminfo_used_vs_baseline_gib": peak_meminfo / GIB,
            "shm_delta_bytes": peak_shm,
            "shm_delta_gib": peak_shm / GIB,
            "tree_proc_count": peak_count,
            "tick_at_peak_rss": asdict(peak_tick) if peak_tick is not None else None,
            "top_processes_at_peak_rss": [asdict(s) for s in peak_samples],
        },
        "outputs": {
            "timeseries_csv": str(timeseries_path),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[profile] peak tree RSS = {peak_rss / GIB:.2f} GiB "
          f"(MemAvail delta peak = {peak_meminfo / GIB:.2f} GiB, "
          f"shm delta peak = {peak_shm / GIB:.2f} GiB, "
          f"max procs in tree = {peak_count})", file=sys.stderr)
    print(f"[profile] summary: {summary_path}", file=sys.stderr)
    print(f"[profile] timeseries: {timeseries_path}", file=sys.stderr)
    return ret if ret is not None else 1


if __name__ == "__main__":
    sys.exit(main())
