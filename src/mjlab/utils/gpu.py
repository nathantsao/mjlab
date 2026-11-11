"""Utilities for GPU selection and management."""

import os
from typing import Literal

import torch

GpuId = int | str


def select_gpus(
  gpu_ids: list[int] | Literal["all"] | None,
  *,
  backend: Literal["cuda", "xpu"] | None = None,
  logical: bool = False,
) -> tuple[list[GpuId] | None, int]:
  """Select GPUs based on device visibility and user specification.

  This function treats the `gpu_ids` parameter as indices into the existing
  CUDA_VISIBLE_DEVICES environment variable for CUDA builds, or ZE_AFFINITY_MASK
  for XPU builds. If the applicable variable is unset, it defaults to all
  devices reported by that backend.

  Args:
    gpu_ids: Either a list of indices into the visibility mask, "all", or None
      for CPU.
    backend: Backend to select from; defaults to CUDA for CUDA builds, else XPU.
    logical: Return logical indices reported by Torch instead of physical mask
      entries. Use this when the existing visibility mask must remain unchanged.

  Returns:
    A tuple of (selected_gpu_ids, num_gpus) where:
    - selected_gpu_ids: Physical GPU IDs (logical indices if requested), or None
    - num_gpus: Number of GPUs selected (0 for CPU mode)

  Examples:
    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    >>> select_gpus([0, 1])
    ([0, 1], 2)

    >>> os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    >>> select_gpus([0])  # Selects physical GPU 1
    ([1], 1)

    >>> select_gpus("all")  # Selects all GPUs in CUDA_VISIBLE_DEVICES
    ([1, 3], 2)

    >>> select_gpus(None)  # CPU mode
    (None, 0)

    >>> os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Empty CUDA_VISIBLE_DEVICES
    >>> select_gpus([0])
    (None, 0)
  """
  # CPU mode requested explicitly.
  if gpu_ids is None:
    return None, 0

  if backend is None:
    backend = "cuda" if torch.backends.cuda.is_built() else "xpu"
  device_module = torch.get_device_module(backend)
  visibility_variable = (
    "CUDA_VISIBLE_DEVICES" if backend == "cuda" else "ZE_AFFINITY_MASK"
  )

  existing_visible_devices = None if logical else os.environ.get(visibility_variable)
  if existing_visible_devices is not None:
    available_gpus: list[GpuId] = [
      int(x.strip()) if x.strip().isdigit() else x.strip()
      for x in existing_visible_devices.split(",")
      if x.strip()
    ]
  else:
    available_gpus: list[GpuId] = list(range(device_module.device_count()))

  if not available_gpus:
    return None, 0

  # Map gpu_ids indices to actual GPU IDs.
  selected: list[GpuId]
  if gpu_ids == "all":
    selected = available_gpus
  else:
    if len(set(gpu_ids)) != len(gpu_ids):
      raise ValueError("--gpu-ids cannot contain duplicate device indices.")
    invalid = [index for index in gpu_ids if not 0 <= index < len(available_gpus)]
    if invalid:
      raise ValueError(
        f"Invalid {backend.upper()} indices {invalid}; "
        f"{len(available_gpus)} device(s) are visible."
      )
    # gpu_ids are indices into available_gpus.
    selected = [available_gpus[i] for i in gpu_ids]

  num_gpus = len(selected)

  return selected, num_gpus
