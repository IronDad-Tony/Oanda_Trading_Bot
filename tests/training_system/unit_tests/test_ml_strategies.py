import sys
import types
from abc import ABCMeta
from unittest.mock import MagicMock, patch, ANY, call
import pytest
import logging
from typing import Dict, Optional, Any, List, Tuple, Union, Callable

# --- Start Comprehensive Torch Mock ---
mock_torch = types.ModuleType('torch')
mock_nn = types.ModuleType('torch.nn')
mock_functional = types.ModuleType('torch.nn.functional')
mock_optim = types.ModuleType('torch.optim')
mock_utils_data = types.ModuleType('torch.utils.data')

class MockDevice:
    def __init__(self, device_str):
        self.type = device_str.split(':')[0] if ':' in device_str else device_str
        self.index = int(device_str.split(':')[1]) if ':' in device_str and self.type not in ['mps', 'cpu'] else None
        if self.type == 'mps' and ':' in device_str:
             self.type = 'mps'
             self.index = None
        if self.type == 'cpu' and ':' in device_str: # e.g. "cpu:0"
            self.type = 'cpu'
            self.index = None


    def __str__(self):
        if self.index is not None and self.type not in ['mps', 'cpu']:
            return f"{self.type}:{self.index}"
        return self.type

    def __repr__(self):
        return f"MockDevice(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, str):
            # Normalize other string for comparison
            other_type = other.split(':')[0] if ':' in other else other
            other_index = int(other.split(':')[1]) if ':' in other and other_type not in ['mps', 'cpu'] else None
            if other_type == 'mps' and ':' in other : other_index = None
            if other_type == 'cpu' and ':' in other : other_index = None
            
            return self.type == other_type and self.index == other_index
        if isinstance(other, MockDevice):
            return self.type == other.type and self.index == other.index
        return False

# Define mock dtypes and assign to mock_torch first
mock_torch.float32 = type('float32', (), {'dtype_str': 'float32'})()
mock_torch.float64 = type('float64', (), {'dtype_str': 'float64'})()
mock_torch.double = mock_torch.float64
mock_torch.float16 = type('float16', (), {'dtype_str': 'float16'})()
mock_torch.half = mock_torch.float16
mock_torch.int64 = type('int64', (), {'dtype_str': 'int64'})()
mock_torch.long = mock_torch.int64
mock_torch.int32 = type('int32', (), {'dtype_str': 'int32'})()
mock_torch.int16 = type('int16', (), {'dtype_str': 'int16'})()
mock_torch.int8 = type('int8', (), {'dtype_str': 'int8'})()
mock_torch.uint8 = type('uint8', (), {'dtype_str': 'uint8'})()
mock_torch.bool = type('bool', (), {'dtype_str': 'bool'})()
mock_torch.complex64 = type('complex64', (), {'dtype_str': 'complex64'})()
mock_torch.complex128 = type('complex128', (), {'dtype_str': 'complex128'})()


class MockTensor:
    def __init__(self, data=None, device=None, dtype=None, requires_grad=False, _shape=None): # Added _shape for internal use
        self.data = data
        if isinstance(device, str): self._device = MockDevice(device)
        elif isinstance(device, MockDevice): self._device = device
        else: self._device = MockDevice('cpu')

        self.dtype = dtype if dtype else mock_torch.float32 # Default to float32 if None
        self.requires_grad = requires_grad
        self.grad = None
        if requires_grad: # Initialize grad if requires_grad is true
            # Simplified grad initialization
            self.grad = MockTensor(data=0, device=self._device, dtype=self.dtype, requires_grad=False)


        if _shape is not None: # Allow explicit shape setting
            self._shape = _shape
        else: # Infer shape
            self._shape = ()
            if hasattr(data, 'shape') and isinstance(data.shape, tuple):
                self._shape = data.shape
            elif isinstance(data, (list, tuple)):
                if not data: self._shape = (0,)
                else:
                    if all(isinstance(x, (int, float, complex, bool)) for x in data):
                        self._shape = (len(data),)
                    elif all(isinstance(x, (list, tuple)) and len(x) == (len(data[0]) if data and hasattr(data[0], '__len__') else 0) for x in data):
                        self._shape = (len(data), len(data[0]) if data and hasattr(data[0], '__len__') else 0)
            elif isinstance(data, (int, float, complex, bool)):
                self._shape = ()
            
            if not self._shape and isinstance(data, (list,tuple)): # Try numpy for shape
                try:
                    import numpy as np
                    arr = np.array(data, dtype=object if not data or not isinstance(data[0], (int,float,bool)) else None) # Handle mixed types better for shape
                    self._shape = arr.shape
                except ImportError: pass
                except Exception: pass # Catch other numpy errors during shape inference


    @property
    def shape(self): return self._shape
    def detach(self): return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=False, _shape=self.shape)
    def cpu(self):
        new_tensor = self.clone() if self._device.type != 'cpu' else self
        new_tensor._device = MockDevice('cpu')
        if new_tensor.grad: new_tensor.grad.cpu()
        return new_tensor
    def cuda(self, device_idx=None):
        dev_str = f'cuda:{device_idx}' if device_idx is not None else 'cuda'
        new_tensor = self.clone() if str(self._device) != dev_str else self
        new_tensor._device = MockDevice(dev_str)
        if new_tensor.grad: new_tensor.grad.cuda(device_idx)
        return new_tensor
        
    def to(self, *args, **kwargs):
        target_device = None
        target_dtype = None
        copy_tensor = kwargs.get('copy', False)
        non_blocking = kwargs.get('non_blocking', False) # Unused in mock but part of sig

        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, (str, MockDevice)): target_device = MockDevice(first_arg) if isinstance(first_arg, str) else first_arg
            elif hasattr(first_arg, 'dtype_str'): target_dtype = first_arg # It's a dtype
            elif isinstance(first_arg, MockTensor): # .to(other_tensor)
                target_device = first_arg.device
                target_dtype = first_arg.dtype
            
            if len(args) > 1 and hasattr(args[1], 'dtype_str'): target_dtype = args[1] # .to(device, dtype)
        
        if 'device' in kwargs and kwargs['device'] is not None: target_device = MockDevice(kwargs['device']) if isinstance(kwargs['device'], str) else kwargs['device']
        if 'dtype' in kwargs and kwargs['dtype'] is not None: target_dtype = kwargs['dtype']

        # If no change, return self unless copy is True
        changed_device = target_device and self._device != target_device
        changed_dtype = target_dtype and self.dtype != target_dtype

        if not copy_tensor and not changed_device and not changed_dtype:
            return self

        new_data = self.data # Simplistic data copy
        new_device = target_device if target_device else self._device
        new_dtype = target_dtype if target_dtype else self.dtype
        
        new_tensor = MockTensor(new_data, device=new_device, dtype=new_dtype, requires_grad=self.requires_grad, _shape=self.shape)
        if self.grad:
            new_tensor.grad = self.grad.to(*args, **kwargs) # Recursively call .to on grad
        return new_tensor

    def item(self):
        if self.numel() == 1:
            if isinstance(self.data, list) and len(self.data) == 1: return self.data[0]
            if isinstance(self.data, tuple) and len(self.data) == 1: return self.data[0]
            if not hasattr(self.data, '__len__'): return self.data # Scalar
            if hasattr(self.data, 'item'): return self.data.item() # Numpy scalar
            if self.shape == () or self.shape == (1,): # More robust for scalar-like
                 # Try to extract single element if data is list/tuple of one
                 if isinstance(self.data, (list,tuple)) and len(self.data) == 1 and isinstance(self.data[0], (int,float,bool)): return self.data[0]
                 return self.data # Fallback
        raise ValueError("only one element tensors can be converted to Python scalars")

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if self.requires_grad and self.grad is None:
            # Simplified grad: tensor of ones-like data
            grad_data_val = 1.0 if self.dtype in [mock_torch.float16, mock_torch.float32, mock_torch.float64] else 1
            try:
                import numpy as np
                grad_data = np.full(self.shape, grad_data_val)
            except ImportError:
                if self.shape == (): grad_data = grad_data_val
                elif len(self.shape) == 1: grad_data = [grad_data_val] * self.shape[0]
                else: grad_data = None # Complex shapes hard without numpy
            self.grad = MockTensor(grad_data, device=self._device, dtype=self.dtype, _shape=self.shape)
        elif self.requires_grad and self.grad is not None and gradient is not None:
             if isinstance(self.grad.data, (int, float)) and isinstance(gradient.data, (int, float)):
                 self.grad.data += gradient.data # Simplified accumulation

    def sum(self, dim=None, keepdim=False):
        sum_data = 0
        if isinstance(self.data, (int, float)): sum_data = self.data
        elif isinstance(self.data, list): sum_data = sum(el for el in self.data if isinstance(el, (int,float)))
        
        new_shape = self.shape
        if dim is None: # Sum all elements
            new_shape = () if not keepdim else tuple([1] * len(self.shape)) if self.shape else ()
        else: # Sum along a dimension
            actual_dim = dim if dim >= 0 else len(self.shape) + dim
            if 0 <= actual_dim < len(self.shape):
                temp_shape = list(self.shape)
                if keepdim: temp_shape[actual_dim] = 1
                else: temp_shape.pop(actual_dim)
                new_shape = tuple(temp_shape) if temp_shape else ()
            else: pass # Invalid dim, shape unchanged for mock
        return MockTensor(sum_data, device=self._device, dtype=self.dtype, _shape=new_shape)

    def mean(self, dim=None, keepdim=False):
        mean_data = 0
        num_elements = self.numel()
        if isinstance(self.data, (int, float)): mean_data = self.data
        elif isinstance(self.data, list) and self.data:
            numeric_elements = [el for el in self.data if isinstance(el, (int,float))]
            if numeric_elements: mean_data = sum(numeric_elements) / len(numeric_elements) if len(numeric_elements) > 0 else 0
        
        # Shape logic similar to sum
        new_shape = self.shape
        if dim is None:
            new_shape = () if not keepdim else tuple([1] * len(self.shape)) if self.shape else ()
        else:
            actual_dim = dim if dim >= 0 else len(self.shape) + dim
            if 0 <= actual_dim < len(self.shape):
                temp_shape = list(self.shape)
                if keepdim: temp_shape[actual_dim] = 1
                else: temp_shape.pop(actual_dim)
                new_shape = tuple(temp_shape) if temp_shape else ()
        return MockTensor(mean_data, device=self._device, dtype=self.dtype, _shape=new_shape)

    def max(self, dim=None, keepdim=False):
        val_data = self.data[0] if isinstance(self.data, list) and self.data else self.data # Simplified
        idx_data = 0 # Simplified
        val_tensor = MockTensor(val_data, device=self._device, dtype=self.dtype)
        idx_tensor = MockTensor(idx_data, device=self._device, dtype=mock_torch.long)

        new_shape = self.shape
        if dim is not None:
            actual_dim = dim if dim >= 0 else len(self.shape) + dim
            if 0 <= actual_dim < len(self.shape):
                temp_shape = list(self.shape)
                if keepdim: temp_shape[actual_dim] = 1
                else: temp_shape.pop(actual_dim)
                new_shape = tuple(temp_shape) if temp_shape else ()
        elif dim is None: # Max over all elements
             new_shape = () if not keepdim else tuple([1] * len(self.shape)) if self.shape else ()
        
        val_tensor._shape = new_shape
        idx_tensor._shape = new_shape # Indices tensor has same shape as output values
        return val_tensor, idx_tensor

    def min(self, dim=None, keepdim=False): return self.max(dim, keepdim) # Simplified, same as max
    
    def argmax(self, dim=None, keepdim=False):
        idx_data = 0 # Simplified
        idx_tensor = MockTensor(idx_data, device=self._device, dtype=mock_torch.long)
        new_shape = self.shape
        if dim is not None:
            actual_dim = dim if dim >= 0 else len(self.shape) + dim
            if 0 <= actual_dim < len(self.shape):
                temp_shape = list(self.shape)
                if keepdim: temp_shape[actual_dim] = 1
                else: temp_shape.pop(actual_dim)
                new_shape = tuple(temp_shape) if temp_shape else ()
        elif dim is None: # Argmax over flattened tensor
            new_shape = () # Scalar index if dim is None and keepdim=False (common case)
            if keepdim: # if keepdim=True for argmax over all, shape is (1,1,...)
                new_shape = tuple([1] * len(self.shape)) if self.shape else ()

        idx_tensor._shape = new_shape
        return idx_tensor

    def unsqueeze(self, dim):
        new_shape_list = list(self.shape)
        actual_dim = dim if dim >= 0 else len(new_shape_list) + dim + 1
        if not (0 <= actual_dim <= len(new_shape_list)): raise IndexError("Dimension out of range")
        new_shape_list.insert(actual_dim, 1)
        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=tuple(new_shape_list))

    def squeeze(self, dim=None):
        new_shape_list = list(self.shape)
        if dim is None:
            squeezed_shape = [s for s in new_shape_list if s != 1]
            # If all dims were 1, torch returns shape (1,) e.g. (1,1,1) -> (1,). (1,) -> ().
            # This is tricky. Let's simplify: if squeezed_shape is empty, it becomes () if original had content, or (0,) if original was (0,)
            if not squeezed_shape:
                 _shape = () if self.shape and self.shape != (0,) else self.shape # if original was (0,) keep (0,)
            else: _shape = tuple(squeezed_shape)

        else:
            actual_dim = dim if dim >= 0 else len(new_shape_list) + dim
            if not (0 <= actual_dim < len(new_shape_list)): raise IndexError("Dimension out of range")
            if new_shape_list[actual_dim] == 1:
                new_shape_list.pop(actual_dim)
            _shape = tuple(new_shape_list) if new_shape_list else () # Handle squeezing last dim of size 1 to scalar
        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=_shape)

    def view(self, *shape_args):
        target_shape_tuple = shape_args[0] if len(shape_args) == 1 and isinstance(shape_args[0], (list, tuple)) else shape_args
        
        # Resolve -1
        numel_self = self.numel()
        prod_known_dims = 1
        minus_one_idx = -1
        for i, s_i in enumerate(target_shape_tuple):
            if s_i == -1:
                if minus_one_idx != -1: raise ValueError("can only specify one unknown dimension")
                minus_one_idx = i
            elif s_i < 0 : raise ValueError("shape entry cannot be negative unless it's -1")
            else: prod_known_dims *= s_i
        
        final_shape_list = list(target_shape_tuple)
        if minus_one_idx != -1:
            if prod_known_dims == 0: # e.g. view(0, -1) from a tensor with 0 elements
                if numel_self == 0: final_shape_list[minus_one_idx] = 0 # or some other size if numel allows
                else: raise RuntimeError(f"shape '{target_shape_tuple}' is invalid for input of size {numel_self}")
            elif numel_self % prod_known_dims != 0:
                raise RuntimeError(f"shape '{target_shape_tuple}' is invalid for input of size {numel_self}")
            final_shape_list[minus_one_idx] = numel_self // prod_known_dims
        
        final_shape = tuple(final_shape_list)
        
        # Check if new numel matches old numel
        new_numel = 1
        for s_f in final_shape: new_numel *= s_f
        if numel_self != new_numel:
            raise RuntimeError(f"shape '{final_shape}' is invalid for input of size {numel_self}")

        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=final_shape)

    def reshape(self, *shape_args): return self.view(*shape_args)
    def permute(self, *dims):
        if len(dims) != len(self.shape): raise ValueError("number of dims does not match tensor dimensions")
        if any(d < 0 or d >= len(self.shape) for d in dims) or len(set(dims)) != len(dims): raise ValueError("invalid dimension permutation")
        new_shape_list = [self.shape[i] for i in dims]
        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=tuple(new_shape_list))

    def transpose(self, dim0, dim1):
        actual_dim0 = dim0 if dim0 >= 0 else len(self.shape) + dim0
        actual_dim1 = dim1 if dim1 >= 0 else len(self.shape) + dim1
        if not (0 <= actual_dim0 < len(self.shape) and 0 <= actual_dim1 < len(self.shape)): raise IndexError("Dimension out of range")
        
        new_shape_list = list(self.shape)
        new_shape_list[actual_dim0], new_shape_list[actual_dim1] = new_shape_list[actual_dim1], new_shape_list[actual_dim0]
        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=tuple(new_shape_list))

    def contiguous(self): return self
    def float(self): return self.to(dtype=mock_torch.float32)
    def long(self): return self.to(dtype=mock_torch.int64)
    def bool(self): return self.to(dtype=mock_torch.bool)
    def double(self): return self.to(dtype=mock_torch.float64)
    def half(self): return self.to(dtype=mock_torch.float16)

    def size(self, dim=None):
        if dim is not None:
            actual_dim = dim if dim >= 0 else len(self.shape) + dim
            if not (0 <= actual_dim < len(self.shape)): raise IndexError("Dimension out of range")
            return self.shape[actual_dim]
        return self.shape

    def dim(self): return len(self.shape)
    def numel(self):
        if not self.shape: return 1 # Scalar
        if 0 in self.shape: return 0
        prod = 1
        for s in self.shape: prod *= s
        return prod

    def __len__(self): return self.shape[0] if self.shape and len(self.shape) > 0 else 0
    def __repr__(self): return f"MockTensor({self.data}, shape={self.shape}, device='{self._device}', dtype={getattr(self.dtype, 'dtype_str', self.dtype)}, requires_grad={self.requires_grad})"

    def _arithmetic_op(self, other, op_name):
        res_data = self.data # Simplified
        res_device = self._device
        res_dtype = self.dtype
        res_shape = self.shape
        
        other_val = other
        if isinstance(other, MockTensor):
            other_val = other.data # Use data for op if other is MockTensor
            # Device/dtype promotion (simplified)
            if other.device.type == 'cuda' and res_device.type != 'cuda': res_device = other.device
            elif other.device.type == 'mps' and res_device.type not in ['cuda']: res_device = other.device
            # Dtype promotion: float > long > int > bool (very simplified)
            if other.dtype == mock_torch.float64 : res_dtype = mock_torch.float64
            elif other.dtype == mock_torch.float32 and res_dtype != mock_torch.float64 : res_dtype = mock_torch.float32
            elif other.dtype == mock_torch.int64 and res_dtype not in [mock_torch.float32, mock_torch.float64]: res_dtype = mock_torch.int64
            # Shape broadcasting (simplified: if other is larger, use its shape)
            if other.numel() > self.numel() and self.numel() == 1 : res_shape = other.shape # Scalar op with tensor
            elif self.numel() > other.numel() and other.numel() == 1: pass # Tensor op with scalar
            elif self.shape != other.shape: pass # Broadcasting needed, not fully mocked for data. Shape remains self.shape or needs broadcasting logic.


        # Actual arithmetic (very simplified, does not change data for mock)
        # if op_name == 'add' and isinstance(self.data, (int,float)) and isinstance(other_val, (int,float)): res_data = self.data + other_val
        # elif op_name == 'sub' and isinstance(self.data, (int,float)) and isinstance(other_val, (int,float)): res_data = self.data - other_val
        # ... etc. For mock, often data is not critical.

        return MockTensor(res_data, device=res_device, dtype=res_dtype, _shape=res_shape,
                          requires_grad=self.requires_grad or (isinstance(other, MockTensor) and other.requires_grad))

    def __add__(self, other): return self._arithmetic_op(other, 'add')
    def __radd__(self, other): return MockTensor(other)._arithmetic_op(self, 'add') if not isinstance(other, MockTensor) else other._arithmetic_op(self, 'add')
    def __sub__(self, other): return self._arithmetic_op(other, 'sub')
    def __rsub__(self, other): return MockTensor(other)._arithmetic_op(self, 'sub') if not isinstance(other, MockTensor) else other._arithmetic_op(self, 'sub') # (other-self)
    def __mul__(self, other): return self._arithmetic_op(other, 'mul')
    def __rmul__(self, other): return MockTensor(other)._arithmetic_op(self, 'mul') if not isinstance(other, MockTensor) else other._arithmetic_op(self, 'mul')
    def __truediv__(self, other): return self._arithmetic_op(other, 'div')
    def __rtruediv__(self, other): return MockTensor(other)._arithmetic_op(self, 'div') if not isinstance(other, MockTensor) else other._arithmetic_op(self, 'div')
    def __pow__(self, other): return self._arithmetic_op(other, 'pow')
    def __rpow__(self, other): return MockTensor(other)._arithmetic_op(self, 'pow') if not isinstance(other, MockTensor) else other._arithmetic_op(self, 'pow')


    def _comparison_op(self, other, op_type):
        # Result is a bool tensor. Shape follows broadcasting rules. Simplified: use self.shape.
        # Data is True/False. For mock, can return a tensor of all Trues.
        return MockTensor(True, device=self._device, dtype=mock_torch.bool, _shape=self.shape)

    def __eq__(self, other): return self._comparison_op(other, 'eq')
    def __ne__(self, other): return self._comparison_op(other, 'ne')
    def __lt__(self, other): return self._comparison_op(other, 'lt')
    def __le__(self, other): return self._comparison_op(other, 'le')
    def __gt__(self, other): return self._comparison_op(other, 'gt')
    def __ge__(self, other): return self._comparison_op(other, 'ge')

    def __getitem__(self, key):
        # Highly simplified. Returns a new tensor, possibly scalar or smaller.
        # Data is not actually sliced. Shape is naively adjusted.
        new_shape = self.shape
        if isinstance(key, int) and self.shape: new_shape = self.shape[1:] if len(self.shape) > 1 else ()
        # More complex slicing (tuples, slices) would need more logic for shape.
        # For now, return a tensor that might represent a slice.
        return MockTensor(self.data, device=self._device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=new_shape)

    def __setitem__(self, key, value): pass # No actual data modification
    @property
    def device(self): return self._device
    def clone(self):
        cloned_tensor = MockTensor(self.data, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad, _shape=self.shape)
        if self.grad: cloned_tensor.grad = self.grad.clone()
        return cloned_tensor
    
    # For matmul, bmm, etc.
    def matmul(self, other_tensor):
        # Simplified: C = A @ B. Shape of C depends on A and B.
        # (n,m) @ (m,p) -> (n,p)
        # (b,n,m) @ (b,m,p) -> (b,n,p)
        s_shape = self.shape
        o_shape = other_tensor.shape
        # Basic 2D matmul
        if len(s_shape) == 2 and len(o_shape) == 2 and s_shape[1] == o_shape[0]:
            res_shape = (s_shape[0], o_shape[1])
        # Basic batch matmul
        elif len(s_shape) == 3 and len(o_shape) == 3 and s_shape[0] == o_shape[0] and s_shape[2] == o_shape[1]:
            res_shape = (s_shape[0], s_shape[1], o_shape[2])
        else: # Fallback or more complex broadcasting
            res_shape = s_shape # Placeholder
        return MockTensor(self.data, device=self.device, dtype=self.dtype, _shape=res_shape) # Data not actually computed
    
    __matmul__ = matmul


class MockParameter(MockTensor):
    def __init__(self, data=None, requires_grad=True, device=None, dtype=None):
        # If data is already a MockTensor, use its attributes for initialization
        _data_val = data.data if isinstance(data, MockTensor) else data
        _device_val = device if device else (data.device if isinstance(data, MockTensor) else None)
        _dtype_val = dtype if dtype else (data.dtype if isinstance(data, MockTensor) else None)
        _shape_val = data.shape if isinstance(data, MockTensor) else None # Pass shape if data is tensor

        super().__init__(_data_val, device=_device_val, dtype=_dtype_val, requires_grad=requires_grad, _shape=_shape_val)

    def __repr__(self):
        return f"MockParameter(data={self.data}, shape={self.shape}, requires_grad={self.requires_grad}, device='{self.device}')"


class MockModule(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parameters = {}
        self._modules = {}
        self._buffers = {} # For register_buffer
        self.training = True
        self.device = mock_torch.device('cpu') # Default device

    def __call__(self, *args, **kwargs):
        if hasattr(self, 'forward'): return self.forward(*args, **kwargs)
        raise NotImplementedError("Forward method not implemented")

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            if module is not None: module.train(mode)
        return self
    def eval(self): return self.train(False)

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters(recurse=recurse): yield param
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        for name, param in self._parameters.items():
            if param is not None: yield prefix + name, param
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_parameters(prefix + name + '.', recurse=True)
    
    def children(self):
        for _, module in self._modules.items():
            if module is not None: yield module
    def modules(self):
        yield self
        for _, module in self._modules.items():
            if module is not None: yield from module.modules()

    def __setattr__(self, name, value):
        if isinstance(value, MockParameter):
            if not hasattr(self, '_parameters'): self._parameters = {}
            self._parameters[name] = value
        elif isinstance(value, MockModule):
            if not hasattr(self, '_modules'): self._modules = {}
            self._modules[name] = value
        else: super().__setattr__(name, value)

    def __getattr__(self, name):
        if hasattr(self, '_parameters') and name in self._parameters: return self._parameters[name]
        if hasattr(self, '_modules') and name in self._modules: return self._modules[name]
        if name in self.__dict__: return self.__dict__[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def add_module(self, name: str, module: Optional['MockModule']):
        if not hasattr(self, '_modules'): self._modules = {}
        if module is not None and not isinstance(module, MockModule):
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        self._modules[name] = module
        return module
    
    def register_parameter(self, name: str, param: Optional[MockParameter]):
        if not hasattr(self, '_parameters'): self._parameters = {}
        if param is not None and not isinstance(param, MockParameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' object to parameter '{name}' (torch.nn.Parameter or None required)")
        self._parameters[name] = param
        if param is not None and param.device != self.device : # Ensure param device matches module device
             param.to(self.device)


    def register_buffer(self, name: str, tensor: Optional[MockTensor], persistent: bool = True):
        if not hasattr(self, '_buffers'): self._buffers = {}
        if tensor is not None and not isinstance(tensor, MockTensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' (torch.Tensor or None required)".format(type(tensor).__name__, name))
        self._buffers[name] = tensor
        if tensor is not None and tensor.device != self.device:
            tensor.to(self.device)


    def to(self, *args, **kwargs):
        target_device = None
        target_dtype = None # Modules don't have a single dtype, but params/buffers do

        if len(args) > 0:
            first_arg = args[0]
            if isinstance(first_arg, (str, MockDevice)): target_device = mock_torch.device(first_arg)
            elif hasattr(first_arg, 'dtype_str'): target_dtype = first_arg # dtype
            elif isinstance(first_arg, MockTensor): # .to(other_tensor)
                target_device = first_arg.device
                target_dtype = first_arg.dtype
            if len(args) > 1 and hasattr(args[1], 'dtype_str'): target_dtype = args[1] # .to(device, dtype)
        
        if 'device' in kwargs and kwargs['device'] is not None: target_device = mock_torch.device(kwargs['device'])
        if 'dtype' in kwargs and kwargs['dtype'] is not None: target_dtype = kwargs['dtype']

        if target_device:
            self.device = target_device
            for p_val in self._parameters.values():
                if p_val is not None: p_val.to(target_device)
            for m_val in self._modules.values():
                if m_val is not None: m_val.to(target_device)
            for b_val in self._buffers.values(): # Move buffers too
                if b_val is not None: b_val.to(target_device)
        
        if target_dtype: # Apply dtype to parameters and buffers
            for p_val in self._parameters.values():
                if p_val is not None: p_val.to(dtype=target_dtype)
            for b_val in self._buffers.values():
                if b_val is not None: b_val.to(dtype=target_dtype)
            # Submodules handle their own dtypes internally if needed
        return self

    def load_state_dict(self, state_dict, strict=True):
        # print(f"Mock {type(self).__name__} loading state_dict. Keys: {list(state_dict.keys())}, Strict: {strict}")
        for name, param_data_tensor in state_dict.items():
            module_ptr = self
            parts = name.split('.')
            param_name = parts[-1]
            
            # Navigate to submodule
            for part_idx in range(len(parts) - 1):
                if hasattr(module_ptr, '_modules') and parts[part_idx] in module_ptr._modules:
                    module_ptr = module_ptr._modules[parts[part_idx]]
                else: # Submodule not found
                    if strict: raise RuntimeError(f"Missing key in state_dict: {name} (submodule {parts[part_idx]} not found)")
                    module_ptr = None; break 
            if module_ptr is None: continue

            # Load parameter or buffer
            target_attr = None
            if hasattr(module_ptr, '_parameters') and param_name in module_ptr._parameters:
                target_attr = module_ptr._parameters[param_name]
            elif hasattr(module_ptr, '_buffers') and param_name in module_ptr._buffers: # Check buffers
                target_attr = module_ptr._buffers[param_name]
            
            if target_attr is not None and isinstance(param_data_tensor, MockTensor):
                # In-place update of data, shape, dtype for the MockTensor/MockParameter
                target_attr.data = param_data_tensor.data
                target_attr._shape = param_data_tensor.shape # Use internal _shape
                target_attr.dtype = param_data_tensor.dtype
                target_attr.requires_grad = param_data_tensor.requires_grad # For parameters
                target_attr.to(self.device) # Ensure it's on the correct device
            elif strict:
                raise RuntimeError(f"Missing key(s) in state_dict: {name} (parameter/buffer {param_name} not found or data type mismatch)")
        return MagicMock() # For <All keys matched successfully> like object

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None: destination = {}
        for name, param in self._parameters.items():
            if param is not None: destination[prefix + name] = param if keep_vars else param.clone().detach() # Detach for safety
        for name, buf in self._buffers.items(): # Include buffers
             if buf is not None: destination[prefix + name] = buf if keep_vars else buf.clone().detach()
        for name, module in self._modules.items():
            if module is not None: module.state_dict(destination, prefix + name + '.', keep_vars)
        return destination

    def apply(self, fn):
        for module in self.children():
            if module is not None: module.apply(fn)
        fn(self)
        return self

    def cuda(self, device=None):
        dev_str = f'cuda:{device}' if isinstance(device, int) else 'cuda'
        return self.to(mock_torch.device(dev_str))
    def cpu(self): return self.to(mock_torch.device('cpu'))
    
    def __repr__(self):
        child_lines = []
        if hasattr(self, '_modules'):
            for name, module in self._modules.items():
                mod_str = repr(module) if module is not None else "None"
                mod_str = '\n'.join(['  ' + line for line in mod_str.split('\n')])
                child_lines.append(f'({name}): {mod_str}')
        main_str = self.__class__.__name__ + '('
        if child_lines: main_str += '\n  ' + '\n  '.join(child_lines) + '\n'
        main_str += ')'
        return main_str

# Assign MockModule as the base for nn.Module
mock_nn.Module = MockModule
mock_nn.Parameter = MockParameter # Also assign Parameter to nn directly

# Specific Layer Mocks
class MockLinear(MockModule):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Use mock_torch.empty or mock_torch.zeros to initialize parameters
        self.weight = MockParameter(mock_torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias: self.bias = MockParameter(mock_torch.empty(out_features, device=device, dtype=dtype))
        else: self.register_parameter('bias', None)
        if device: self.to(device) # Ensure module itself is on device

    def forward(self, input_tensor: MockTensor) -> MockTensor:
        batch_size = input_tensor.shape[0] if input_tensor.shape and len(input_tensor.shape)>1 else 1
        # Output shape (batch_size, out_features) or (out_features) if input is (in_features)
        out_shape = (batch_size, self.out_features) if len(input_tensor.shape) > 1 else (self.out_features,)
        if not input_tensor.shape : out_shape = (self.out_features,) # scalar input -> (out_features)
        if len(input_tensor.shape) ==1 and input_tensor.shape[0] == self.in_features: out_shape = (self.out_features,)


        return mock_torch.empty(*out_shape, device=self.weight.device, dtype=self.weight.dtype)

class MockConv2d(MockModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels; self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding # Can also be string e.g. 'same'
        self.dilation = dilation; self.groups = groups; self.padding_mode = padding_mode
        
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.weight = MockParameter(mock_torch.empty(*weight_shape, device=device, dtype=dtype))
        if bias: self.bias = MockParameter(mock_torch.empty(out_channels, device=device, dtype=dtype))
        else: self.register_parameter('bias', None)
        if device: self.to(device)

    def forward(self, input_tensor: MockTensor) -> MockTensor:
        bs, _, h_in, w_in = input_tensor.shape
        # Simplified output H, W calculation
        h_out = (h_in + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1) - 1) // self.stride[0] + 1 if isinstance(self.padding, tuple) else h_in 
        w_out = (w_in + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1) - 1) // self.stride[1] + 1 if isinstance(self.padding, tuple) else w_in
        if isinstance(self.padding, str): # Handle 'same', 'valid' string padding by not changing H,W for mock
            h_out, w_out = h_in, w_in

        return mock_torch.empty(bs, self.out_channels, h_out, w_out, device=input_tensor.device, dtype=input_tensor.dtype)

class MockReLU(MockModule):
    def __init__(self, inplace=False): super().__init__(); self.inplace = inplace
    def forward(self, x: MockTensor): return x # Simplified

class MockSequential(MockModule):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict): # OrderedDict
            for key, module in args[0].items(): self.add_module(key, module)
        else:
            for idx, module in enumerate(args): self.add_module(str(idx), module)
    def forward(self, input_tensor: MockTensor):
        for module in self._modules.values():
            if module is not None: input_tensor = module(input_tensor)
        return input_tensor

class MockDropout(MockModule):
    def __init__(self, p=0.5, inplace=False): super().__init__(); self.p = p; self.inplace = inplace
    def forward(self, x: MockTensor): return x

class MockBatchNormNd(MockModule): # Base for BatchNorm1d, 2d, 3d
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = MockParameter(mock_torch.ones(num_features, device=device, dtype=dtype))
            self.bias = MockParameter(mock_torch.zeros(num_features, device=device, dtype=dtype))
        if self.track_running_stats:
            self.register_buffer('running_mean', mock_torch.zeros(num_features, device=device, dtype=dtype))
            self.register_buffer('running_var', mock_torch.ones(num_features, device=device, dtype=dtype))
            self.register_buffer('num_batches_tracked', mock_torch.tensor(0, dtype=mock_torch.long, device=device))
        if device: self.to(device)
    def forward(self, x: MockTensor): return x

class MockBatchNorm1d(MockBatchNormNd): pass
class MockBatchNorm2d(MockBatchNormNd): pass
class MockBatchNorm3d(MockBatchNormNd): pass


class MockLayerNorm(MockModule):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = MockParameter(mock_torch.ones(*self.normalized_shape, device=device, dtype=dtype))
            self.bias = MockParameter(mock_torch.zeros(*self.normalized_shape, device=device, dtype=dtype))
        if device: self.to(device)
    def forward(self, x: MockTensor): return x

class MockEmbedding(MockModule):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = MockParameter(mock_torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        if device: self.to(device)
    def forward(self, input_tensor: MockTensor) -> MockTensor:
        out_shape = input_tensor.shape + (self.embedding_dim,)
        return mock_torch.empty(*out_shape, device=self.weight.device, dtype=self.weight.dtype)

class MockLSTM(MockModule): # Simplified LSTM
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0., bidirectional=False, device=None, dtype=None):
        super().__init__()
        self.input_size = input_size; self.hidden_size = hidden_size; self.num_layers = num_layers
        self.batch_first = batch_first; self.bidirectional = bidirectional
        # Mock some parameters for state_dict
        num_directions = 2 if bidirectional else 1
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = f'_l{layer}' + (f'_reverse' if direction == 1 and bidirectional else '')
                actual_input_size = input_size if layer == 0 else hidden_size * num_directions
                self.register_parameter(f'weight_ih{suffix}', MockParameter(mock_torch.empty(4*hidden_size, actual_input_size, device=device, dtype=dtype)))
                self.register_parameter(f'weight_hh{suffix}', MockParameter(mock_torch.empty(4*hidden_size, hidden_size, device=device, dtype=dtype)))
                if bias:
                    self.register_parameter(f'bias_ih{suffix}', MockParameter(mock_torch.empty(4*hidden_size, device=device, dtype=dtype)))
                    self.register_parameter(f'bias_hh{suffix}', MockParameter(mock_torch.empty(4*hidden_size, device=device, dtype=dtype)))
        if device: self.to(device)

    def forward(self, input_tensor: MockTensor, hx: Optional[Tuple[MockTensor, MockTensor]] = None) -> Tuple[MockTensor, Tuple[MockTensor, MockTensor]]:
        seq_len_dim, batch_dim = (1,0) if self.batch_first else (0,1)
        batch_size = input_tensor.shape[batch_dim]
        seq_len = input_tensor.shape[seq_len_dim]
        num_directions = 2 if self.bidirectional else 1
        
        out_shape = (batch_size, seq_len, num_directions * self.hidden_size) if self.batch_first else (seq_len, batch_size, num_directions * self.hidden_size)
        output = mock_torch.empty(*out_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        
        hn_shape = (self.num_layers * num_directions, batch_size, self.hidden_size)
        hn = mock_torch.empty(*hn_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        cn = mock_torch.empty(*hn_shape, device=input_tensor.device, dtype=input_tensor.dtype)
        return output, (hn, cn)

mock_nn.Linear = MockLinear
mock_nn.Conv2d = MockConv2d
mock_nn.ReLU = MockReLU
mock_nn.Sequential = MockSequential
mock_nn.Dropout = MockDropout
mock_nn.BatchNorm1d = MockBatchNorm1d
mock_nn.BatchNorm2d = MockBatchNorm2d
mock_nn.BatchNorm3d = MockBatchNorm3d
mock_nn.LayerNorm = MockLayerNorm
mock_nn.Embedding = MockEmbedding
mock_nn.LSTM = MockLSTM
mock_nn.ModuleList = lambda modules=None: list(modules) if modules else [] # Simplified
mock_nn.Identity = type('Identity', (MockModule,), {'forward': lambda self, x: x})


# Populate mock_torch with functions and classes
mock_torch.device = MockDevice
mock_torch.Tensor = MockTensor
mock_torch.Parameter = MockParameter # Also available as torch.Parameter

def _mock_tensor_creation_func(value, *shape, device=None, dtype=None, requires_grad=False, **kwargs):
    # Shape can be specified as varargs or a tuple/list as first arg
    if shape and isinstance(shape[0], (list, tuple)): final_shape = shape[0]
    else: final_shape = shape
    
    # If data is given directly (e.g. torch.tensor([1,2,3]))
    if not final_shape and (isinstance(value, (list,tuple)) or hasattr(value, 'shape')): # value is data
        data = value
        _shape = None # Infer from data
    else: # value is fill_value, shape is specified
        data = value # This is fill_value, actual data array not created for mock
        _shape = final_shape

    return MockTensor(data, device=device, dtype=dtype, requires_grad=requires_grad, _shape=_shape)

mock_torch.empty = lambda *shape, **kwargs: _mock_tensor_creation_func(None, *shape, **kwargs) # Data is undefined
mock_torch.zeros = lambda *shape, **kwargs: _mock_tensor_creation_func(0, *shape, **kwargs)
mock_torch.ones = lambda *shape, **kwargs: _mock_tensor_creation_func(1, *shape, **kwargs)
mock_torch.full = lambda shape, fill_value, **kwargs: _mock_tensor_creation_func(fill_value, shape, **kwargs)
mock_torch.randn = lambda *shape, **kwargs: _mock_tensor_creation_func(0.0, *shape, **kwargs) # Data is random normal
mock_torch.rand = lambda *shape, **kwargs: _mock_tensor_creation_func(0.0, *shape, **kwargs) # Data is random uniform
mock_torch.tensor = lambda data, **kwargs: _mock_tensor_creation_func(data, **kwargs) # Shape inferred from data

mock_torch.is_tensor = lambda obj: isinstance(obj, MockTensor)
mock_torch.is_floating_point = lambda tensor: tensor.dtype in [mock_torch.float32, mock_torch.float16, mock_torch.float64, mock_torch.complex64, mock_torch.complex128]
mock_torch.get_default_dtype = lambda: mock_torch.float32
mock_torch.set_default_dtype = MagicMock()
mock_torch.no_grad = lambda: MagicMock(__enter__=lambda: None, __exit__=lambda a,b,c: None)
mock_torch.manual_seed = MagicMock()
mock_torch.save = MagicMock()
mock_torch.load = MagicMock(return_value={}) # Return empty dict for state_dict
mock_torch.compile = MagicMock(return_value=lambda x: x) # Pass-through decorator

# Mock torch.cuda
mock_torch.cuda = types.ModuleType('torch.cuda')
mock_torch.cuda.is_available = MagicMock(return_value=False)
mock_torch.cuda.device_count = MagicMock(return_value=0)
mock_torch.cuda.current_device = MagicMock(return_value=0)
mock_torch.cuda.get_device_name = MagicMock(return_value="MockCUDA Device")
mock_torch.cuda.manual_seed = MagicMock()
mock_torch.cuda.manual_seed_all = MagicMock()
mock_torch.cuda.empty_cache = MagicMock()

# Mock torch.backends
mock_torch.backends = types.ModuleType('torch.backends')
mock_torch.backends.mps = types.ModuleType('torch.backends.mps')
mock_torch.backends.mps.is_available = MagicMock(return_value=False)
mock_torch.backends.mps.is_built = MagicMock(return_value=False)

# Mock functional helpers
mock_functional.relu = lambda x: x
mock_functional.softmax = lambda x, dim=None, dtype=None: x # dtype arg added in later torch versions
mock_functional.log_softmax = lambda x, dim=None, dtype=None: x
mock_functional.sigmoid = lambda x: x
mock_functional.tanh = lambda x: x
mock_functional.mse_loss = lambda i, t, reduction='mean': mock_torch.tensor(0.0)
mock_functional.cross_entropy = lambda i, t, reduction='mean', ignore_index=-100: mock_torch.tensor(0.0)
mock_functional.binary_cross_entropy_with_logits = lambda i, t, reduction='mean': mock_torch.tensor(0.0)
mock_functional.dropout = lambda x, p=0.5, training=False, inplace=False: x
mock_functional.layer_norm = lambda x, normalized_shape, weight=None, bias=None, eps=1e-5: x
mock_functional.embedding = lambda input, weight, padding_idx=None: mock_torch.empty(*input.shape, weight.shape[1], device=weight.device, dtype=weight.dtype)


# Mock optimizers
class MockOptimizer:
    def __init__(self, params, lr=0.001, *args, **kwargs):
        self.param_groups = [{'params': list(params), 'lr': lr}] # Simplified
        self.state = {} # For state_dict
    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none: p.grad = None
                    else: p.grad.data = 0 # Mock zeroing grad data
    def step(self): pass
    def state_dict(self): return {'state': self.state, 'param_groups': self.param_groups}
    def load_state_dict(self, state_dict): self.state = state_dict.get('state', {}); self.param_groups = state_dict.get('param_groups', [])

mock_optim.Adam = MockOptimizer
mock_optim.AdamW = MockOptimizer
mock_optim.SGD = MockOptimizer
mock_optim.RMSprop = MockOptimizer

# Mock Schedulers
class MockLambdaLR:
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer; self.lr_lambda = lr_lambda; self.last_epoch = last_epoch
    def step(self): self.last_epoch +=1
    def state_dict(self): return {'last_epoch': self.last_epoch}
    def load_state_dict(self, state_dict): self.last_epoch = state_dict.get('last_epoch', -1)

mock_optim.lr_scheduler = types.ModuleType('torch.optim.lr_scheduler')
mock_optim.lr_scheduler.LambdaLR = MockLambdaLR

# Mock Dataset and DataLoader
class MockDataset:
    def __init__(self, data=None): self.data = data if data is not None else []
    def __getitem__(self, index): return self.data[index]
    def __len__(self): return len(self.data)

class MockDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, *args, **kwargs):
        self.dataset = dataset; self.batch_size = batch_size; self.shuffle = shuffle
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[i : i + self.batch_size] # Simplified batching
    def __len__(self): return (len(self.dataset) + self.batch_size - 1) // self.batch_size

mock_utils_data.Dataset = MockDataset
mock_utils_data.DataLoader = MockDataLoader
mock_utils_data.TensorDataset = lambda *tensors: MockDataset(list(zip(*[t.data if isinstance(t, MockTensor) else t for t in tensors])))


# Assign submodules to mock_torch
mock_torch.nn = mock_nn
mock_torch.nn.functional = mock_functional
mock_torch.optim = mock_optim
mock_torch.utils = types.ModuleType('torch.utils') # Create parent torch.utils
mock_torch.utils.data = mock_utils_data


# Apply mocks to sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_nn
sys.modules['torch.nn.functional'] = mock_functional
sys.modules['torch.optim'] = mock_optim
sys.modules['torch.optim.lr_scheduler'] = mock_optim.lr_scheduler # Ensure this is also patched
sys.modules['torch.utils.data'] = mock_utils_data
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.backends'] = mock_torch.backends
sys.modules['torch.backends.mps'] = mock_torch.backends.mps

# --- End Comprehensive Torch Mock ---

# The rest of your test file starts here
import unittest
# ... (other imports for the actual test file like pandas, specific project modules)

# If pandas is used in this file, import it here after the mock
try:
    import pandas as pd
except ImportError:
    pd = None 

# Import project-specific modules AFTER the mock is in place
# from src.agent.strategies.ml_strategies import LSTMStrategy, TransformerStrategy # etc.
# from src.agent.strategies.base_strategy import StrategyConfig, BaseStrategy
# from src.data_handling.data_handler_arrow import ArrowDataHandler 

# Placeholder for the rest of the file:
# print("Torch mock applied. Rest of test_ml_strategies.py should follow.")