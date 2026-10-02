from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Select CUDA, then Intel XPU, then CPU, or validate an explicit device."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError(
            "device must be 'auto', 'cpu', 'cuda[:index]', or 'xpu[:index]'."
        )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build."
        )
    if resolved.type == "xpu" and not torch.xpu.is_available():
        raise RuntimeError(
            "Intel XPU is not available. Install PyTorch with XPU support from "
            "https://download.pytorch.org/whl/xpu and check your Intel GPU driver."
        )
    if resolved.type != "cpu" and resolved.index is not None:
        count = getattr(torch, resolved.type).device_count()
        if resolved.index >= count:
            raise ValueError(
                f"Device {resolved} does not exist; found {count} devices."
            )
    return resolved
