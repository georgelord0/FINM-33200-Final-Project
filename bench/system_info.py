"""Runtime system information for the generated report."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_system_info() -> dict[str, Any]:
    """Collect reproducibility-relevant hardware and software details."""
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": _os_info(),
        "python": _python_info(),
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "gpu": _gpu_info(),
        "packages": _package_info(),
    }


def write_system_info(path: str | Path) -> dict[str, Any]:
    info = collect_system_info()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    return info


def _os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
    }


def _cpu_info() -> dict[str, Any]:
    return {
        "brand": _cpu_brand(),
        "processor": platform.processor() or platform.machine(),
        "physical_cores": _physical_cpu_count(),
        "logical_cores": os.cpu_count(),
    }


def _memory_info() -> dict[str, Any]:
    total = _total_memory_bytes()
    return {
        "total_gb": round(total / 1024**3, 2) if total else None,
    }


def _gpu_info() -> dict[str, Any]:
    return {
        "nvidia_smi": _nvidia_smi_info(),
        "torch": _torch_info(),
    }


def _package_info() -> dict[str, str | None]:
    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "torch",
        "chronos-forecasting",
        "timesfm",
        "jax",
        "jaxlib",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _physical_cpu_count() -> int | None:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _total_memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass

    if platform.system() == "Windows":
        return _windows_total_memory_bytes()
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _windows_total_memory_bytes() -> int | None:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys)
    return None


def _nvidia_smi_info() -> list[dict[str, Any]]:
    fields = ["name", "memory.total", "driver_version", "compute_cap"]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return []

    rows = []
    for line in result.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(
            {
                "name": values[0],
                "display_name": _gpu_display_name(values[0]),
                "memory_total_mb": _to_int(values[1]),
                "memory_total_gb": _mb_to_gb(_to_int(values[1])),
                "driver_version": values[2],
                "compute_capability": values[3],
                "architecture": _gpu_architecture(values[0], values[3]),
            }
        )
    return rows


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"available": False, "import_error": str(exc)}

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "compiled_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "arch_list": list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "display_name": _gpu_display_name(torch.cuda.get_device_name(idx)),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total_gb": round(props.total_memory / 1024**3, 2),
                    "architecture": _gpu_architecture(
                        torch.cuda.get_device_name(idx),
                        f"{props.major}.{props.minor}",
                    ),
                }
            )
    return info


def _cpu_brand() -> str | None:
    system = platform.system()
    if system == "Windows":
        value = _run_first_line([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ])
        if value:
            return value
        value = _run_first_line(["wmic", "cpu", "get", "Name", "/value"])
        if value and "=" in value:
            return value.split("=", 1)[1].strip()
    elif system == "Darwin":
        return _run_first_line(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _run_first_line(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _gpu_display_name(name: str) -> str:
    match = re.search(r"\bRTX\s+\d{3,5}(?:\s+(?:Ti|SUPER))?\b", name, flags=re.I)
    if match:
        return match.group(0).upper().replace("  ", " ")
    cleaned = re.sub(r"^NVIDIA\s+", "", name, flags=re.I)
    cleaned = re.sub(r"^GEFORCE\s+", "", cleaned, flags=re.I)
    return cleaned.strip() or name


def _gpu_architecture(name: str, compute_capability: str | None) -> str | None:
    if "RTX 50" in name.upper() or compute_capability == "12.0":
        return "Blackwell"
    if compute_capability and compute_capability.startswith("9."):
        return "Hopper"
    if compute_capability == "8.9":
        return "Ada Lovelace"
    if compute_capability and compute_capability.startswith("8."):
        return "Ampere"
    return None


def _mb_to_gb(value: int | None) -> float | None:
    return round(value / 1024, 2) if value is not None else None


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write report runtime system info")
    parser.add_argument("--output", type=Path, default=Path("results/system_info.json"))
    args = parser.parse_args()
    info = write_system_info(args.output)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
