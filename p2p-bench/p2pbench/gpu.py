"""GPU discovery and P2P verification.

Everything here is best-effort and degrades gracefully when a tool is missing,
so the harness can still run (with warnings) on partial environments.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout: {' '.join(cmd)}"


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


@dataclass
class Gpu:
    index: int
    name: str
    bus_id: str
    uuid: str


def list_gpus() -> list[Gpu]:
    rc, out, _ = _run([
        "nvidia-smi",
        "--query-gpu=index,name,pci.bus_id,uuid",
        "--format=csv,noheader",
    ])
    gpus: list[Gpu] = []
    if rc != 0:
        return gpus
    for line in out.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) >= 4:
            gpus.append(Gpu(int(cols[0]), cols[1], cols[2], cols[3]))
    return gpus


def verify_device_order(
    expected_indices: tuple[int, ...] = (1, 2),
    expected_substring: str = "5060 Ti",
) -> dict:
    """Confirm that CUDA indices used in the vllm command map to the expected
    cards. With CUDA_DEVICE_ORDER=PCI_BUS_ID the nvidia-smi index order matches
    what vLLM sees via CUDA_VISIBLE_DEVICES."""
    gpus = {g.index: g for g in list_gpus()}
    result = {"ok": True, "checked": [], "all_gpus": [asdict(g) for g in gpus.values()]}
    for idx in expected_indices:
        g = gpus.get(idx)
        ok = g is not None and expected_substring.lower() in g.name.lower()
        result["checked"].append({
            "index": idx,
            "name": g.name if g else None,
            "bus_id": g.bus_id if g else None,
            "matches": ok,
        })
        result["ok"] = result["ok"] and ok
    return result


def topo_matrix() -> str:
    rc, out, err = _run(["nvidia-smi", "topo", "-m"])
    return out if rc == 0 else f"[topo -m failed] {err}"


def topo_p2p() -> str:
    # Newer nvidia-smi: P2P read/write capability matrix.
    rc, out, err = _run(["nvidia-smi", "topo", "-p2p", "rw"])
    return out if rc == 0 else f"[topo -p2p failed] {err}"


def driver_cuda_versions() -> dict:
    info: dict = {}
    rc, out, _ = _run([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ])
    info["driver_version"] = out.strip().splitlines()[0] if rc == 0 and out.strip() else None

    rc, out, _ = _run(["nvcc", "--version"])
    info["nvcc"] = out.strip() if rc == 0 else None

    try:
        with open("/proc/driver/nvidia/version") as fh:
            info["proc_driver"] = fh.read().strip()
    except OSError:
        info["proc_driver"] = None

    try:
        import torch  # noqa: PLC0415
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["torch_nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version()) \
            if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    return info


def run_cuda_samples_p2p(binary_path: str | None) -> str:
    """Run the official cuda-samples p2pBandwidthLatencyTest if available.

    This is the authoritative P2P proof: compare the P2P=Disabled vs P2P=Enabled
    bandwidth matrices. If they are roughly equal, P2P is NOT actually engaging
    and the whole comparison is invalid (common on consumer Blackwell over PCIe
    without the open-driver P2P patch)."""
    if not binary_path:
        return "[skipped] no --p2p-test-bin provided"
    rc, out, err = _run([binary_path], timeout=300)
    return out if rc == 0 else f"[p2p test failed rc={rc}] {err}\n{out}"


def torch_p2p_probe(dev_a: int = 0, dev_b: int = 1, mb: int = 256) -> dict:
    """Portable fallback P2P probe using PyTorch.

    Reports cudaCanAccessPeer and a crude device-to-device copy bandwidth. If
    peer access is unavailable the copy is staged through host and bandwidth is
    markedly lower -- a useful sanity signal even without cuda-samples."""
    try:
        import time
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available() or torch.cuda.device_count() <= max(dev_a, dev_b):
            return {"ok": False, "reason": "insufficient visible CUDA devices"}
        can = bool(torch.cuda.can_device_access_peer(dev_a, dev_b))
        n = mb * 1024 * 1024 // 4
        a = torch.empty(n, dtype=torch.float32, device=f"cuda:{dev_a}")
        b = torch.empty(n, dtype=torch.float32, device=f"cuda:{dev_b}")
        for _ in range(3):
            b.copy_(a); torch.cuda.synchronize(dev_b)
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            b.copy_(a)
        torch.cuda.synchronize(dev_b)
        dt = time.perf_counter() - t0
        gbps = (mb / 1024.0) * iters / dt
        return {"ok": True, "can_access_peer": can,
                "d2d_copy_GBps": round(gbps, 2), "size_MB": mb}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}


def gpu_memory_used_mb(indices: tuple[int, ...] | list[int]) -> dict:
    """Return {index: MiB used} for the given GPU indices (None if unknown)."""
    rc, out, _ = _run([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ])
    used: dict[int, float] = {}
    if rc == 0:
        for line in out.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 2:
                try:
                    used[int(cols[0])] = float(cols[1])
                except ValueError:
                    continue
    return {i: used.get(i) for i in indices}


def wait_for_vram_free(indices: tuple[int, ...] | list[int],
                       threshold_mb: float = 1024, timeout: float = 180,
                       poll: float = 2.0) -> bool:
    """Block until every listed GPU reports < threshold_mb used, or timeout.
    Closes the gap between a model process exiting and the driver actually
    reclaiming VRAM, which is what causes OOM on the next scenario's load."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        used = gpu_memory_used_mb(indices)
        vals = [v for v in used.values() if v is not None]
        if vals and all(v <= threshold_mb for v in vals):
            return True
        time.sleep(poll)
    return False


def collect_p2p_evidence(p2p_test_bin: str | None,
                         expected_indices: tuple[int, ...]) -> dict:
    return {
        "device_order": verify_device_order(expected_indices),
        "topo_matrix": topo_matrix(),
        "topo_p2p": topo_p2p(),
        "cuda_samples_p2p": run_cuda_samples_p2p(p2p_test_bin),
        "torch_probe": torch_p2p_probe(),
        "versions": driver_cuda_versions(),
    }
