"""Device selection and ordered, zero-copy Torch/Warp execution."""

import torch
import warp as wp


def resolve_device(device: str | torch.device | None = None) -> str:
  """Resolve a Torch device, preferring CUDA, then XPU, then CPU.

  Explicit accelerator requests fail rather than falling back to CPU. Warp's
  ``sycl:N`` identifiers are not Torch device identifiers.
  """
  if device is None:
    if torch.cuda.is_available():
      device = "cuda:0"
    elif torch.xpu.is_available():
      device = "xpu:0"
    else:
      device = "cpu"
  selected = torch.device(device)
  if selected.type == "cpu":
    return "cpu"
  if selected.type not in ("cuda", "xpu"):
    raise ValueError(f"Unsupported device {selected}; expected cpu, cuda:N, or xpu:N.")
  backend = torch.get_device_module(selected)
  if not backend.is_available():
    raise RuntimeError(f"Requested {selected.type.upper()} device is not available.")
  index = backend.current_device() if selected.index is None else selected.index
  if not 0 <= index < backend.device_count():
    raise ValueError(f"Device index {index} is out of range for {selected.type}.")
  return f"{selected.type}:{index}"


def get_warp_device(device: str | torch.device) -> wp.Device:
  """Resolve the Warp device and share Torch's current queue on XPU."""
  wp.init()
  wp.set_torch_stream_enabled(device)
  return wp.device_from_torch(torch.device(device))
