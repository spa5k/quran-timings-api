from __future__ import annotations

import shutil
import subprocess
from typing import Any


def doctor_gpu() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "nvidia_smi": None,
        "recommendation": "Install CUDA-enabled PyTorch (cu128) for strict GPU accuracy mode.",
    }

    try:
        import torch  # type: ignore

        report["torch_available"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

        devices: list[dict[str, Any]] = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_gb": round(float(props.total_memory) / (1024**3), 2),
                        "major": int(props.major),
                        "minor": int(props.minor),
                    }
                )
        report["devices"] = devices
    except Exception as exc:
        report["torch_error"] = str(exc)

    if shutil.which("nvidia-smi") is not None:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            report["nvidia_smi"] = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        else:
            report["nvidia_smi_error"] = proc.stderr.strip()

    return report
