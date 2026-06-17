"""Capture host/kernel state relevant to GPU P2P reproducibility."""

from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else f"[rc={p.returncode}] {p.stderr}\n{p.stdout}"
    except FileNotFoundError:
        return f"[not found] {cmd[0]}"
    except subprocess.TimeoutExpired:
        return f"[timeout] {' '.join(cmd)}"


def capture_kernel_cmdline() -> dict:
    out = {}
    try:
        out["proc_cmdline"] = Path("/proc/cmdline").read_text().strip()
    except OSError as e:
        out["proc_cmdline"] = f"[error] {e}"
    grub = Path("/etc/default/grub")
    if grub.exists():
        lines = [ln for ln in grub.read_text().splitlines()
                 if ln.strip().startswith(("GRUB_CMDLINE_LINUX", "GRUB_CMDLINE_LINUX_DEFAULT"))]
        out["etc_default_grub"] = "\n".join(lines) if lines else "[no GRUB_CMDLINE_* lines]"
    else:
        out["etc_default_grub"] = "[/etc/default/grub not present]"
    return out


def capture_kernel_modules() -> dict:
    lsmod = _run(["lsmod"])
    nvidia_lines = "\n".join(
        ln for ln in lsmod.splitlines()
        if ln.startswith(("nvidia", "nvidia_", "nvidia-")) or ln.split(" ")[0].startswith("nvidia")
    )
    peermem_loaded = any(ln.split(" ")[0] == "nvidia_peermem" for ln in lsmod.splitlines())
    return {
        "lsmod_nvidia": nvidia_lines or "[no nvidia modules listed]",
        "nvidia_peermem_loaded": peermem_loaded,
        "modinfo_nvidia": _run(["modinfo", "nvidia"]),
        "lsmod_full": lsmod,
    }


def capture_sysctl(keys: list[str] | None = None) -> str:
    keys = keys or ["vm.nr_hugepages", "kernel.numa_balancing",
                    "vm.zone_reclaim_mode", "kernel.yama.ptrace_scope"]
    return "\n".join(f"{k} = {_run(['sysctl', '-n', k]).strip()}" for k in keys)


def _grep(text: str, needles: tuple[str, ...]) -> str:
    out = [ln for ln in text.splitlines()
           if any(n.lower() in ln.lower() for n in needles)]
    return "\n".join(out) if out else "[no matching lines]"


def capture_dmesg() -> str:
    raw = _run(["dmesg", "-T"])
    if raw.startswith("[rc=") or raw.startswith("[not found]"):
        # dmesg often needs root; fall back to kernel journal.
        raw = _run(["journalctl", "-k", "-b", "0", "--no-pager"])
    return _grep(raw, ("nvidia", "nvrm", "xid", "pcie", "gpu", "aer"))


def capture_journal_window(since: _dt.datetime, until: _dt.datetime) -> str:
    fmt = "%Y-%m-%d %H:%M:%S"
    raw = _run(["journalctl", "-k", "--no-pager",
                "--since", since.strftime(fmt), "--until", until.strftime(fmt)])
    return _grep(raw, ("nvidia", "nvrm", "xid", "pcie", "gpu", "aer"))


def write_environment(out_dir: str | Path) -> dict:
    """Write all static environment artifacts and return a summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmdline = capture_kernel_cmdline()
    modules = capture_kernel_modules()
    sysctl = capture_sysctl()
    dmesg = capture_dmesg()

    (out_dir / "kernel_cmdline.txt").write_text(
        f"# /proc/cmdline\n{cmdline['proc_cmdline']}\n\n"
        f"# /etc/default/grub\n{cmdline['etc_default_grub']}\n")
    (out_dir / "kernel_modules.txt").write_text(
        f"# nvidia_peermem loaded: {modules['nvidia_peermem_loaded']}\n\n"
        f"# nvidia modules\n{modules['lsmod_nvidia']}\n\n"
        f"# modinfo nvidia\n{modules['modinfo_nvidia']}\n")
    (out_dir / "sysctl.txt").write_text(sysctl + "\n")
    (out_dir / "dmesg_gpu.txt").write_text(dmesg + "\n")
    (out_dir / "lsmod_full.txt").write_text(modules["lsmod_full"] + "\n")

    return {
        "proc_cmdline": cmdline["proc_cmdline"],
        "nvidia_peermem_loaded": modules["nvidia_peermem_loaded"],
    }
