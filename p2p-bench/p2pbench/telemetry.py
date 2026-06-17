"""Background telemetry sampler.

Samples PCIe link gen/width, temperature, utilisation, memory and power for the
target GPUs once per second, tagging every row with a wall-clock timestamp so it
can be correlated against the benchmark window (and used for tokens/joule).
"""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path

FIELDS = [
    "timestamp",
    "index",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
]


class TelemetrySampler:
    def __init__(self, csv_path: str | Path, indices: list[int] | None = None,
                 interval: float = 1.0):
        self.csv_path = Path(csv_path)
        self.indices = indices
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _query(self) -> list[list[str]]:
        cmd = ["nvidia-smi",
               f"--query-gpu={','.join(FIELDS)}",
               "--format=csv,noheader,nounits"]
        if self.indices:
            cmd.insert(1, "-i")
            cmd.insert(2, ",".join(str(i) for i in self.indices))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if p.returncode != 0:
                return []
            rows = []
            wall = time.time()
            for line in p.stdout.strip().splitlines():
                cols = [c.strip() for c in line.split(",")]
                # Replace nvidia-smi's own timestamp with our monotonic wall time
                cols[0] = f"{wall:.3f}"
                rows.append(cols)
            return rows
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def _loop(self):
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["wall_epoch"] + FIELDS[1:])
            while not self._stop.is_set():
                t0 = time.time()
                for row in self._query():
                    writer.writerow(row)
                fh.flush()
                dt = self.interval - (time.time() - t0)
                if dt > 0:
                    self._stop.wait(dt)

    def __enter__(self) -> "TelemetrySampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def mean_power_in_window(csv_path: str | Path, start: float, end: float,
                         indices: list[int] | None = None) -> float | None:
    """Average summed-across-GPU power draw (W) within [start, end] epoch range.
    Used to convert energy into tokens/joule."""
    try:
        rows_by_t: dict[float, float] = {}
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                t = float(r["wall_epoch"])
                if not (start <= t <= end):
                    continue
                if indices is not None and int(r["index"]) not in indices:
                    continue
                try:
                    rows_by_t[t] = rows_by_t.get(t, 0.0) + float(r["power.draw"])
                except (ValueError, KeyError):
                    continue
        if not rows_by_t:
            return None
        return sum(rows_by_t.values()) / len(rows_by_t)
    except OSError:
        return None
