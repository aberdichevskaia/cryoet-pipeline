from __future__ import annotations

from enum import StrEnum
from typing import Any


class DevicePreference(StrEnum):
    """User-facing compute device preference."""

    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def normalize_device(value: str | DevicePreference) -> DevicePreference:
    """Normalize CLI/config device values."""

    try:
        return DevicePreference(str(value).lower())
    except ValueError as exc:
        allowed = ", ".join(device.value for device in DevicePreference)
        raise ValueError(f"unsupported device {value!r}; expected one of: {allowed}") from exc


def resolve_device(
    preference: str | DevicePreference = DevicePreference.AUTO,
    *,
    torch_module: Any | None = None,
) -> DevicePreference:
    """Resolve `auto` to cuda, mps, or cpu.

    `torch_module` is injectable so tests can exercise the decision tree without
    requiring CUDA, Apple Silicon, or PyTorch to be installed.
    """

    normalized = normalize_device(preference)
    if normalized is not DevicePreference.AUTO:
        return normalized

    torch = torch_module if torch_module is not None else _try_import_torch()
    if torch is None:
        return DevicePreference.CPU

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        if cuda.is_available():
            return DevicePreference.CUDA

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is not None and callable(getattr(mps, "is_available", None)):
        if mps.is_available():
            return DevicePreference.MPS

    return DevicePreference.CPU


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
