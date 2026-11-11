"""Bridge for seamless PyTorch-Warp interoperability with zero-copy memory sharing.

Provides automatic wrapping of Warp arrays as PyTorch-compatible objects while
preserving shared memory and GPU graph compatibility.
"""

from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import torch
import warp as wp

T = TypeVar("T")


class TorchArray:
  """Warp array that behaves like a torch.Tensor with shared memory."""

  def __init__(self, wp_array: wp.array, nworld: int | None = None) -> None:
    """Initialize the tensor proxy with a Warp array."""
    device = wp_array.device
    if device is None:
      raise ValueError("TorchArray requires an allocated Warp array with a device.")
    if device.is_sycl:
      wp.set_torch_stream_enabled(wp.device_to_torch(device))
    self._wp_array = wp_array
    # Workaround for Warp bug: wp.to_torch fails on empty CPU arrays.
    if device.is_cpu and wp_array.size == 0:
      scalar_type = getattr(wp_array.dtype, "_wp_scalar_type_", wp_array.dtype)
      shape = (*wp_array.shape, *getattr(wp_array.dtype, "_shape_", ()))
      scalar_type = getattr(wp_array.dtype, "_wp_scalar_type_", wp_array.dtype)
      shape = (*wp_array.shape, *getattr(wp_array.dtype, "_shape_", ()))
      self._torch_tensor = torch.zeros(shape, dtype=wp.dtype_to_torch(scalar_type))
    else:
      self._torch_tensor = wp.to_torch(wp_array)

    if (
      nworld is not None
      and nworld > 1
      and len(self._torch_tensor.shape) > 0
      and self._torch_tensor.shape[0] == 1
      and (self._torch_tensor.stride(0) == 0 or getattr(wp_array, "_is_batched", False))
    ):
      new_shape = (nworld,) + self._torch_tensor.shape[1:]
      self._torch_tensor = self._torch_tensor.expand(new_shape)

  @property
  def _tensor(self) -> torch.Tensor:
    return self._torch_tensor

  @property
  def wp_array(self) -> wp.array:
    return self._wp_array

  def __repr__(self) -> str:
    """Return string representation of the underlying tensor."""
    return repr(self._tensor)

  def __getitem__(self, idx: Any) -> Any:
    """Get item(s) from the tensor using standard indexing."""
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any) -> None:
    """Set item(s) in the tensor using standard indexing."""
    self._torch_tensor[idx] = value

  def __getattr__(self, name: str) -> Any:
    """Delegate attribute access to the underlying tensor."""
    return getattr(self._tensor, name)

  @classmethod
  def __torch_function__(
    cls,
    func: Any,
    types: Tuple[type, ...],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
  ) -> Any:
    """Intercept torch.* function calls to unwrap TorchArray objects."""
    if kwargs is None:
      kwargs = {}

    # Only intercept when at least one argument is our proxy.
    if not any(issubclass(t, cls) for t in types):
      return NotImplemented

    def _unwrap(x: Any) -> Any:
      """Unwrap TorchArray objects to their underlying tensors."""
      return x._tensor if isinstance(x, cls) else x

    # Unwrap all TorchArray objects in args and kwargs.
    unwrapped_args = tuple(_unwrap(arg) for arg in args)
    unwrapped_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}

    return func(*unwrapped_args, **unwrapped_kwargs)

  # Arithmetic operators.

  def __add__(self, other: Any) -> Any:
    return self._tensor + other

  def __radd__(self, other: Any) -> Any:
    return other + self._tensor

  def __sub__(self, other: Any) -> Any:
    return self._tensor - other

  def __rsub__(self, other: Any) -> Any:
    return other - self._tensor

  def __mul__(self, other: Any) -> Any:
    return self._tensor * other

  def __rmul__(self, other: Any) -> Any:
    return other * self._tensor

  def __truediv__(self, other: Any) -> Any:
    return self._tensor / other

  def __rtruediv__(self, other: Any) -> Any:
    return other / self._tensor

  def __pow__(self, other: Any) -> Any:
    return self._tensor**other

  def __rpow__(self, other: Any) -> Any:
    return other**self._tensor

  def __neg__(self) -> Any:
    return -self._tensor

  def __pos__(self) -> Any:
    return +self._tensor

  def __abs__(self) -> Any:
    return abs(self._tensor)

  # Comparison operators.

  def __eq__(self, other: Any) -> Any:
    return self._tensor == other

  def __ne__(self, other: Any) -> Any:
    return self._tensor != other

  def __lt__(self, other: Any) -> Any:
    return self._tensor < other

  def __le__(self, other: Any) -> Any:
    return self._tensor <= other

  def __gt__(self, other: Any) -> Any:
    return self._tensor > other

  def __ge__(self, other: Any) -> Any:
    return self._tensor >= other


def _contains_warp_arrays(obj: Any) -> bool:
  """Check if an object or its attributes contain any Warp arrays."""
  if isinstance(obj, wp.array):
    return True

  # Check if it's a struct-like object with attributes
  if hasattr(obj, "__dict__"):
    return any(
      isinstance(getattr(obj, attr), wp.array)
      for attr in dir(obj)
      if not attr.startswith("_")
    )

  return False


class WarpBridge(Generic[T]):
  """Wraps mjwarp objects to expose Warp arrays as PyTorch tensors.

  Automatically converts Warp array attributes to TorchArray objects
  on access, enabling direct PyTorch operations on simulation data.
  Recursively wraps nested structures that contain Warp arrays.

  IMPORTANT: This wrapper is read-only. To modify array data, use
  in-place operations like `obj.field[:] = value`. Direct assignment
  like `obj.field = new_array` will raise an AttributeError to prevent
  accidental memory address changes that break GPU graphs.
  """

  def __init__(self, struct: T, nworld: int | None = None) -> None:
    object.__setattr__(self, "_struct", struct)
    object.__setattr__(self, "_wrapped_cache", {})
    object.__setattr__(self, "_nworld", nworld)

  def __getattr__(self, name: str) -> Any:
    """Get attribute from the wrapped data, wrapping Warp arrays as TorchArray."""
    # Check cache first to avoid recreating wrappers.
    if name in self._wrapped_cache:
      return self._wrapped_cache[name]

    val = getattr(self._struct, name)

    # Wrap Warp arrays.
    if isinstance(val, wp.array):
      wrapped = TorchArray(val, nworld=self._nworld)
      self._wrapped_cache[name] = wrapped
      return wrapped

    # Recursively wrap nested structures that contain Warp arrays.
    if _contains_warp_arrays(val):
      wrapped = WarpBridge(val, nworld=self._nworld)
      self._wrapped_cache[name] = wrapped
      return wrapped

    return val

  def __setattr__(self, name: str, value: Any) -> None:
    """Prevent attribute setting to maintain GPU graph safety."""
    raise AttributeError(
      f"Cannot set attribute '{name}' on WarpBridge. "
      f"This wrapper is read-only to preserve memory addresses for GPU graphs. "
      f"Use in-place operations instead: obj.{name}[:] = value"
    )

  def __repr__(self) -> str:
    """Return string representation of the wrapped struct."""
    return f"WarpBridge({repr(self._struct)})"

  @property
  def struct(self) -> T:
    """Access the underlying wrapped struct."""
    return self._struct

  def clear_cache(self) -> None:
    """Clear the wrapped cache to force re-wrapping of arrays.

    This should be called after operations that modify the underlying warp arrays,
    such as expand_model_fields(), to ensure the cache reflects the updated arrays.
    """
    object.__setattr__(self, "_wrapped_cache", {})
