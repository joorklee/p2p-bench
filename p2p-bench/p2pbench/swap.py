"""llama-swap lifecycle and model selection.

The harness selects a scenario by issuing a tiny OpenAI-compatible request whose
``model`` field equals the scenario id; llama-swap then (un)loads the matching
vLLM instance. Coupling is intentionally minimal so this works across llama-swap
versions. The unload endpoint is configurable because its path has changed
between releases.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlamaSwap:
    def __init__(self, binary: str, config_path: str | Path, port: int = 8080,
                 log_path: str | Path | None = None):
        self.binary = binary
        self.config_path = str(config_path)
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.log_path = Path(log_path) if log_path else None
        self.proc: subprocess.Popen | None = None
        self._log_fh = None

    # ---- process lifecycle ----
    def start(self) -> None:
        cmd = [self.binary, "--config", self.config_path, "--listen", f":{self.port}"]
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "w")
            self.proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)
        else:
            self.proc = subprocess.Popen(cmd)
        self._wait_listening(timeout=30)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log_fh:
            self._log_fh.close()

    def __enter__(self) -> "LlamaSwap":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- http helpers ----
    def _get(self, path: str, timeout: float = 5) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(self.base + path, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return 0, ""

    def _post(self, path: str, payload: bytes, timeout: float = 600) -> tuple[int, str]:
        req = urllib.request.Request(self.base + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return 0, str(e)

    def _wait_listening(self, timeout: int) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            code, _ = self._get("/v1/models", timeout=2)
            if code:
                return
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-swap exited during startup; check its log")
            time.sleep(1)
        raise TimeoutError("llama-swap did not start listening in time")

    # ---- model selection ----
    def load_and_wait(self, model_id: str, timeout: int = 1200) -> tuple[bool, str]:
        """Trigger a model load via a 1-token request; wait until it succeeds.
        Returns (ok, message). On failure, message is the upstream error text."""
        payload = (
            '{"model":"%s","prompt":"ping","max_tokens":1,"temperature":0}' % model_id
        ).encode()
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            code, body = self._post("/v1/completions", payload, timeout=min(120, timeout))
            if code == 200:
                return True, "ok"
            last = f"http {code}: {body[:500]}"
            # 503/5xx while the model boots is expected; keep polling.
            time.sleep(3)
        return False, last or "timeout waiting for model load"

    def unload(self) -> bool:
        """Free VRAM now instead of waiting for ttl. Tries the current
        versioned endpoint first, then legacy paths. Unloads all models (only
        one runs at a time here)."""
        attempts = [
            ("POST", "/api/models/unload"),   # current llama-swap
            ("POST", "/unload"),              # legacy (#58)
            ("GET", "/unload"),               # older variant
        ]
        for method, path in attempts:
            if method == "POST":
                code, _ = self._post(path, b"", timeout=30)
            else:
                code, _ = self._get(path, timeout=30)
            if code in (200, 204):
                return True
        return False
