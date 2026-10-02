"""dots.tts package."""

from __future__ import annotations


def _check_torch_install() -> None:
    """Fail fast on torch/torchaudio install problems.

    Runs at package import so users get an actionable message instead of a
    cryptic ``libcudart.so.X: cannot open shared object file`` from
    torchaudio's C extension loader, or a downstream dtype mismatch after a
    silent CPU fallback. Reads versions via ``importlib.metadata`` to avoid
    triggering the broken library load itself.
    """
    import importlib.metadata as _md

    try:
        torch_version = _md.version("torch")
        torchaudio_version = _md.version("torchaudio")
    except _md.PackageNotFoundError:
        return

    _install_hint = (
        "Install matching torch + torchaudio versions for your CPU, CUDA, "
        "or Intel XPU device using the PyTorch install matrix: "
        "https://pytorch.org/get-started/locally/\n"
        "e.g. for CUDA 12.8:\n"
        "  pip install torch==2.8.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cu128\n"
        "For Intel XPU:\n"
        "  pip install torch torchaudio "
        "--index-url https://download.pytorch.org/whl/xpu"
    )

    torch_minor = tuple(torch_version.split("+")[0].split(".")[:2])
    torchaudio_minor = tuple(torchaudio_version.split("+")[0].split(".")[:2])
    if torch_minor != torchaudio_minor:
        raise RuntimeError(
            f"torch ({torch_version}) and torchaudio ({torchaudio_version}) "
            f"minor versions do not match.\n{_install_hint}"
        )

    try:
        import torchaudio  # noqa: F401
    except OSError as exc:
        if "libcudart" in str(exc) or "libcuda" in str(exc):
            import torch  # deferred; only reached if torchaudio failed

            raise RuntimeError(
                f"torchaudio failed to load its native extension: {exc}\n"
                f"Your torch/torchaudio wheels were built for CUDA "
                f"{torch.version.cuda!r}, but the required runtime library is "
                f"not present on this machine.\n{_install_hint}"
            ) from exc
        raise


_check_torch_install()
del _check_torch_install
