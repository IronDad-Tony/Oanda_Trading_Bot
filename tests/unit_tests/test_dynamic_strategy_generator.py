# Mocking imports at the very top
import sys
import types
import unittest
from unittest.mock import patch, MagicMock, ANY, call
import numpy as np
import pandas as pd
import copy
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Type
from abc import ABCMeta # Added ABCMeta

# --- BEGIN COMPREHENSIVE TORCH MOCKING ---
# This block must run BEFORE any 'src' module that imports 'torch' is loaded.

class MockTensor:
    def __init__(self, data, dtype=None, device=None):
        self.data = data
        self._is_mps = False
        self.dtype = dtype
        self._device = device if device else MockDevice('cpu')
        self.requires_grad = False # Default for tensors

    def to(self, device_or_dtype, non_blocking=False): # Added non_blocking
        if isinstance(device_or_dtype, MockDevice):
            self._device = device_or_dtype
        elif isinstance(device_or_dtype, types.SimpleNamespace): # Assuming dtypes are SimpleNamespace
            self.dtype = device_or_dtype
        elif isinstance(device_or_dtype, MockTensor): # For tensor.to(other_tensor)
            self.dtype = device_or_dtype.dtype
            self._device = device_or_dtype.device
        return self

    def float(self): self.dtype = mock_torch.float32; return self
    def long(self): self.dtype = mock_torch.int64; return self
    def bool(self): self.dtype = mock_torch.bool; return self
    def detach(self): new_tensor = self.clone(); new_tensor.requires_grad = False; return new_tensor
    def cpu(self): self._device = MockDevice('cpu'); return self
    def numpy(self): return np.array(self.data) if self.data is not None else np.array([])
    def clone(self): new_tensor = copy.deepcopy(self); new_tensor._device = self._device; return new_tensor
    def zero_(self):
        if isinstance(self.data, (list, np.ndarray)):
            self.data = np.zeros_like(self.data).tolist()
        elif isinstance(self.data, (int, float)):
            self.data = 0
        return self
    def __getitem__(self, item):
        if isinstance(item, MockTensor): return self.data[item.data] if self.data is not None else self
        return self.data[item] if self.data is not None else self
    def __setitem__(self, key, value):
        if self.data is not None: self.data[key] = value

    def __add__(self, other): return MockTensor(self.data + (other.data if isinstance(other, MockTensor) else other), dtype=self.dtype, device=self._device)
    def __radd__(self, other): return MockTensor((other.data if isinstance(other, MockTensor) else other) + self.data, dtype=self.dtype, device=self._device)
    def __sub__(self, other): return MockTensor(self.data - (other.data if isinstance(other, MockTensor) else other), dtype=self.dtype, device=self._device)
    def __mul__(self, other): return MockTensor(self.data * (other.data if isinstance(other, MockTensor) else other), dtype=self.dtype, device=self._device)
    def __rmul__(self, other): return MockTensor((other.data if isinstance(other, MockTensor) else other) * self.data, dtype=self.dtype, device=self._device)
    def __truediv__(self, other): return MockTensor(self.data / (other.data if isinstance(other, MockTensor) else other), dtype=self.dtype, device=self._device)
    def __pow__(self, power): return MockTensor(self.data ** (power.data if isinstance(power, MockTensor) else power), dtype=self.dtype, device=self._device)
    def __lt__(self, other): return MockTensor(self.data < (other.data if isinstance(other, MockTensor) else other), dtype=mock_torch.bool, device=self._device)
    def __gt__(self, other): return MockTensor(self.data > (other.data if isinstance(other, MockTensor) else other), dtype=mock_torch.bool, device=self._device)
    def __eq__(self, other):
        if isinstance(other, MockTensor): return MockTensor(np.array_equal(self.data, other.data), dtype=mock_torch.bool, device=self._device)
        return MockTensor(np.array_equal(self.data, other), dtype=mock_torch.bool, device=self._device)
    def __ne__(self, other):
        if isinstance(other, MockTensor): return MockTensor(not np.array_equal(self.data, other.data), dtype=mock_torch.bool, device=self._device)
        return MockTensor(not np.array_equal(self.data, other), dtype=mock_torch.bool, device=self._device)

    def mean(self, dim=None, keepdim=False): # Added dim, keepdim
        res = np.mean(self.data, axis=dim)
        if keepdim and dim is not None:
            res = np.expand_dims(res, axis=dim)
        return MockTensor(res if self.data else 0, dtype=self.dtype, device=self._device)

    def sum(self, dim=None, keepdim=False, dtype=None): # Added dim, keepdim, dtype
        res = np.sum(self.data, axis=dim)
        if keepdim and dim is not None:
            res = np.expand_dims(res, axis=dim)
        return MockTensor(res if self.data else 0, dtype=dtype if dtype else self.dtype, device=self._device)

    def sqrt(self): return MockTensor(np.sqrt(self.data), dtype=self.dtype, device=self._device)
    def exp(self): return MockTensor(np.exp(self.data), dtype=self.dtype, device=self._device)
    def abs(self): return MockTensor(np.abs(self.data), dtype=self.dtype, device=self._device)
    def is_cuda(self): return self._device.type == 'cuda'
    @property
    def shape(self): return np.array(self.data).shape if self.data is not None else (0,)
    def size(self, dim=None):
        s = np.array(self.data).shape if self.data is not None else tuple()
        if dim is None: return s
        return s[dim] if dim < len(s) else 1 # Match PyTorch behavior for out-of-bounds dim
    def unsqueeze(self, dim): return self # Simplified
    def squeeze(self, dim=None): return self # Simplified
    def view(self, *shape): return self # Simplified
    def permute(self, *dims): return self # Simplified
    def transpose(self, dim0, dim1): return self # Simplified
    def contiguous(self): return self # Simplified
    def fill_(self, value):
        if self.data is not None: self.data = np.full_like(self.data, value).tolist()
        return self
    @property
    def device(self): return self._device
    def item(self): return self.data[0] if isinstance(self.data, list) and len(self.data)==1 and isinstance(self.data[0], (int, float, bool)) else self.data # More robust item
    def argmax(self, dim=None, keepdim=False): return MockTensor(np.argmax(self.data, axis=dim), device=self._device) # Added keepdim
    def max(self, dim=None, keepdim=False): # Added keepdim
        if dim is None: return MockTensor(np.max(self.data), device=self._device)
        else:
            max_val = np.max(self.data, axis=dim)
            argmax_val = np.argmax(self.data, axis=dim)
            if keepdim:
                max_val = np.expand_dims(max_val, axis=dim)
                argmax_val = np.expand_dims(argmax_val, axis=dim)
            return MockTensor(max_val, device=self._device), MockTensor(argmax_val, device=self._device)
    def min(self, dim=None, keepdim=False): # Added min, keepdim
        if dim is None: return MockTensor(np.min(self.data), device=self._device)
        else:
            min_val = np.min(self.data, axis=dim)
            argmin_val = np.argmin(self.data, axis=dim)
            if keepdim:
                min_val = np.expand_dims(min_val, axis=dim)
                argmin_val = np.expand_dims(argmin_val, axis=dim)
            return MockTensor(min_val, device=self._device), MockTensor(argmin_val, device=self._device)

    def backward(self, gradient=None, retain_graph=None, create_graph=False): pass
    @property
    def grad(self):
        if not hasattr(self, '_grad'): self._grad = None
        return self._grad
    @grad.setter
    def grad(self, value): self._grad = value
    def requires_grad_(self, requires_grad=True): self.requires_grad = requires_grad; return self
    def numel(self): return np.prod(self.shape) # Added numel

class MockParameter(MockTensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data) # Device will be cpu by default from MockTensor
        self.requires_grad = requires_grad

class MockDevice:
    def __init__(self, device_type):
        self.type = device_type
    def __str__(self): return self.type
    def __repr__(self): return f"device(type='{self.type}')"

# Using ABCMeta for MockModule to avoid metaclass conflicts if BaseStrategy uses ABC
class MockModule(metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        self._parameters: Dict[str, MockParameter] = {}
        self._modules: Dict[str, 'MockModule'] = {}
        self._buffers: Dict[str, MockTensor] = {}
        self._is_mock_module = True
        self._is_mps = False
        self._device = MockDevice('cpu')
        self.training = True

    def to(self, device_or_dtype_or_tensor, non_blocking=False):
        target_device = None
        target_dtype = None

        if isinstance(device_or_dtype_or_tensor, MockDevice):
            target_device = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, str): # e.g. "cuda"
             target_device = MockDevice(device_or_dtype_or_tensor)
        elif isinstance(device_or_dtype_or_tensor, types.SimpleNamespace): # dtype
            target_dtype = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, MockTensor): # tensor.to(other_tensor)
            target_device = device_or_dtype_or_tensor.device
            target_dtype = device_or_dtype_or_tensor.dtype

        if target_device:
            self._device = target_device
            for p in self._parameters.values(): p.to(target_device)
            for m in self._modules.values(): m.to(target_device)
            for b in self._buffers.values(): b.to(target_device)
        
        if target_dtype: # Apply dtype to parameters and buffers
            for p in self._parameters.values(): p.to(target_dtype)
            for b in self._buffers.values(): b.to(target_dtype)
        return self

    def train(self, mode=True): self.training = mode; return self
    def eval(self): self.training = False; return self

    def parameters(self, recurse: bool = True) -> List[MockParameter]:
        params_list = list(self._parameters.values())
        if recurse:
            for m in self._modules.values():
                params_list.extend(m.parameters(True))
        return params_list

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> List[Tuple[str, MockParameter]]:
        named_params_list = []
        for name, p in self._parameters.items():
            named_params_list.append((prefix + name if prefix else name, p))
        if recurse:
            for name, m in self._modules.items():
                sub_prefix = prefix + name + '.' if prefix else name + '.'
                named_params_list.extend(m.named_parameters(sub_prefix, True))
        return named_params_list
        
    def children(self) -> List['MockModule']: return list(self._modules.values())
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs): raise NotImplementedError
    def apply(self, fn):
        fn(self)
        for m in self._modules.values(): m.apply(fn)
        return self
    def state_dict(self, destination=None, prefix='', keep_vars=False): return {} # Simplified
    def load_state_dict(self, state_dict, strict=True): pass
    def cuda(self, device=None): self.to(MockDevice('cuda')); return self
    def cpu(self): self.to(MockDevice('cpu')); return self
    def add_module(self, name: str, module: Optional['MockModule']):
        if module is None: self._modules.pop(name, None)
        else: self._modules[name] = module
    def register_parameter(self, name: str, param: Optional[MockParameter]):
        if param is None: self._parameters.pop(name, None)
        else: self._parameters[name] = param
    def register_buffer(self, name: str, tensor: Optional[MockTensor], persistent: bool = True):
        if tensor is None: self._buffers.pop(name, None)
        else: self._buffers[name] = tensor

    def __setattr__(self, name, value):
        if isinstance(value, MockParameter): self.register_parameter(name, value)
        elif isinstance(value, MockModule): self._modules[name] = value
        elif isinstance(value, MockTensor):
            if name in self._buffers: self._buffers[name] = value
            else: super().__setattr__(name, value)
        else: super().__setattr__(name, value)

mock_torch = types.ModuleType('torch')
mock_torch_nn = types.ModuleType('torch.nn')
mock_torch_nn.Module = MockModule
mock_torch_nn.Parameter = MockParameter
# Add common nn layers (can be simple MockModule instances or more specific mocks if needed)
mock_torch_nn.Linear = lambda in_features, out_features, bias=True, device=None, dtype=None: MockModule()
mock_torch_nn.Conv1d = lambda in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None: MockModule()
mock_torch_nn.ReLU = lambda inplace=False: MockModule()
mock_torch_nn.Sigmoid = lambda: MockModule()
mock_torch_nn.Tanh = lambda: MockModule()
mock_torch_nn.Softmax = lambda dim=None: MockModule()
mock_torch_nn.Dropout = lambda p=0.5, inplace=False: MockModule()
mock_torch_nn.BatchNorm1d = lambda num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None: MockModule()
mock_torch_nn.LayerNorm = lambda normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None: MockModule()
mock_torch_nn.Embedding = lambda num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None: MockModule()
mock_torch_nn.LSTM = lambda *args, **kwargs: MockModule()
mock_torch_nn.GRU = lambda *args, **kwargs: MockModule()
mock_torch_nn.TransformerEncoderLayer = lambda d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05, batch_first=False, norm_first=False, device=None, dtype=None: MockModule()
mock_torch_nn.TransformerEncoder = lambda encoder_layer, num_layers, norm=None, enable_nested_tensor=True: MockModule()
mock_torch_nn.Sequential = lambda *args: MockModule() # Args are modules
mock_torch_nn.ModuleList = lambda modules=None: MockModule() # modules is an iterable of Module
mock_torch_nn.Identity = lambda *args, **kwargs: MockModule()
mock_torch_nn.CrossEntropyLoss = lambda weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0: MockModule()
mock_torch_nn.MSELoss = lambda size_average=None, reduce=None, reduction='mean': MockModule()

mock_torch_nn_functional = types.ModuleType('torch.nn.functional')
mock_torch_nn_functional.relu = lambda x, inplace=False: x
mock_torch_nn_functional.softmax = lambda x, dim=None, _stacklevel=3, dtype=None: x
mock_torch_nn_functional.sigmoid = lambda x: x # This should operate on a MockTensor and return one
mock_torch_nn_functional.tanh = lambda x: x # This should operate on a MockTensor and return one
mock_torch_nn_functional.dropout = lambda x, p=0.5, training=False, inplace=False: x
mock_torch_nn_functional.layer_norm = lambda x, normalized_shape, weight=None, bias=None, eps=1e-05: x
mock_torch_nn_functional.cross_entropy = lambda input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0: MockTensor([0.0])
mock_torch_nn_functional.mse_loss = lambda input, target, size_average=None, reduce=None, reduction='mean': MockTensor([0.0])
mock_torch_nn_functional.gelu = lambda x, approximate='none': x
mock_torch_nn_functional.adaptive_avg_pool2d = lambda x, output_size: x
mock_torch_nn_functional.gumbel_softmax = MagicMock(name="MockGumbelSoftmax", return_value=MockTensor([0.1, 0.9])) # Ensure it returns a MockTensor
mock_torch_nn.functional = mock_torch_nn_functional
mock_torch.nn = mock_torch_nn

mock_torch_optim = types.ModuleType('torch.optim')
class MockOptimizer:
    def __init__(self, params, lr=0.001, **kwargs): # params can be an iterator
        self.param_groups = [{'params': list(params), 'lr': lr, **kwargs}]
        self.state = {} 
    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'grad') and p.grad is not None:
                    if set_to_none: p.grad = None
                    else:
                        if hasattr(p.grad, 'detach'): p.grad = p.grad.detach()
                        p.grad.zero_()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None: loss = closure()
        return loss
mock_torch_optim.Adam = MockOptimizer
mock_torch_optim.SGD = MockOptimizer
mock_torch_optim.AdamW = MockOptimizer
mock_torch.optim = mock_torch_optim

mock_torch_cuda = types.ModuleType('torch.cuda')
mock_torch_cuda.is_available = lambda: False
mock_torch_cuda.device_count = lambda: 0
mock_torch_cuda.current_device = lambda: -1 # Or raise error
mock_torch_cuda.get_device_name = lambda device=None: ""
mock_torch.cuda = mock_torch_cuda

mock_torch_backends = types.ModuleType('torch.backends')
mock_torch_backends_mps = types.ModuleType('torch.backends.mps')
mock_torch_backends_mps.is_available = lambda: False
mock_torch_backends_mps.is_built = lambda: False # Added is_built
mock_torch_backends.mps = mock_torch_backends_mps
mock_torch.backends = mock_torch_backends

mock_torch.Tensor = MockTensor
mock_torch.tensor = lambda data, dtype=None, device=None, requires_grad=False: MockTensor(data, dtype=dtype, device=device)
mock_torch.is_tensor = lambda obj: isinstance(obj, MockTensor)
mock_torch.from_numpy = lambda ndarray: MockTensor(ndarray.tolist()) # Simplified
mock_torch.zeros = lambda *size, out=None, dtype=None, layout=None, device=None, requires_grad=False: MockTensor(np.zeros(size).tolist(), dtype=dtype, device=device)
mock_torch.ones = lambda *size, out=None, dtype=None, layout=None, device=None, requires_grad=False: MockTensor(np.ones(size).tolist(), dtype=dtype, device=device)
mock_torch.randn = lambda *size, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad=False: MockTensor(np.random.randn(*size).tolist(), dtype=dtype, device=device)
mock_torch.empty = lambda *size, out=None, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None: MockTensor(np.empty(size).tolist(), dtype=dtype, device=device)
mock_torch.arange = lambda *args, out=None, dtype=None, layout=None, device=None, requires_grad=False: MockTensor(np.arange(*args).tolist(), dtype=dtype, device=device)
mock_torch.manual_seed = lambda seed: None
mock_torch.set_grad_enabled = lambda mode: None
mock_torch.no_grad = lambda: unittest.mock.MagicMock() # Context manager
mock_torch.save = lambda obj, f, pickle_module=None, pickle_protocol=2, _use_new_zipfile_serialization=True: None
mock_torch.load = lambda f, map_location=None, pickle_module=None, **pickle_load_args: {} # Simplified
mock_torch.float32 = types.SimpleNamespace()
mock_torch.float = mock_torch.float32
mock_torch.int64 = types.SimpleNamespace()
mock_torch.long = mock_torch.int64
mock_torch.bool = types.SimpleNamespace()
mock_torch.bfloat16 = types.SimpleNamespace()
mock_torch.get_default_dtype = lambda: mock_torch.float32
mock_torch.set_default_dtype = lambda d: None
# Top-level functions that operate on tensors or numbers
mock_torch.sigmoid = lambda input, out=None: input.sigmoid() if isinstance(input, MockTensor) else MockTensor(1 / (1 + np.exp(-input)))
mock_torch.tanh = lambda input, out=None: input.tanh() if isinstance(input, MockTensor) else MockTensor(np.tanh(input))
mock_torch.exp = lambda input, out=None: input.exp() if isinstance(input, MockTensor) else MockTensor(np.exp(input))
mock_torch.sqrt = lambda input, out=None: input.sqrt() if isinstance(input, MockTensor) else MockTensor(np.sqrt(input)) # Added torch.sqrt

DEVICE = MockDevice('cpu')
# Make mock_torch.device more flexible:
# if called with a string, returns a new MockDevice
# if called with no args, or with a MockDevice, could return that or a default
# For simplicity, let's have it always return a new MockDevice based on input string, or default 'cpu'
def device_constructor(device_str=None):
    if device_str:
        return MockDevice(device_str)
    return DEVICE # Default 'cpu' device
mock_torch.device = MagicMock(side_effect=device_constructor)


sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['torch.optim'] = mock_torch.optim
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.backends'] = mock_torch.backends
sys.modules['torch.backends.mps'] = mock_torch.backends.mps
# --- END COMPREHENSIVE TORCH MOCKING ---

import torch # This will now import the mock_torch object

# Now import the modules to be tested, AFTER mocks are in sys.modules
from src.agent.enhanced_quantum_strategy_layer import DynamicStrategyGenerator
from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig # Assuming this is correctly importable
from src.agent.optimizers.genetic_optimizer import GeneticOptimizer
# from src.agent.optimizers.bayesian_optimizer import BayesianOptimizer # Keep commented if not essential

# Setup a logger for the test module (can be specific or general)
mock_dsg_logger = logging.getLogger("MockDSG_TestFile")
mock_dsg_logger.addHandler(logging.NullHandler()) # Avoids "no handler" warnings if not configured elsewhere

# Mock StrategyConfig for tests
class MockStrategyConfig(StrategyConfig):
    def __init__(self, name="MockConfig", description="Mock Description", 
                 default_params: Optional[Dict[str, Any]] = None, 
                 strategy_specific_params: Optional[Dict[str, Any]] = None,
                 applicable_assets: Optional[List[str]] = None,
                 **kwargs):
        # Ensure all base StrategyConfig args are passed or defaulted
        super().__init__(
            name=name, 
            description=description,
            default_params=default_params if default_params is not None else {},
            strategy_specific_params=strategy_specific_params if strategy_specific_params is not None else {},
            applicable_assets=applicable_assets if applicable_assets is not None else [],
            **kwargs
        )

    def copy(self): # Ensure copy method exists and works
        return MockStrategyConfig(
            name=self.name, 
            description=self.description,
            default_params=copy.deepcopy(self.default_params),
            strategy_specific_params=copy.deepcopy(self.strategy_specific_params),
            applicable_assets=copy.deepcopy(self.applicable_assets),
            **self.kwargs # Pass along any other kwargs
            )

# Mock strategy classes for testing
class MockStrategy(BaseStrategy):
    def __init__(self, config: Optional[StrategyConfig] = None, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        # Ensure a valid config is passed to BaseStrategy
        final_config = config if config else self.default_config()
        super().__init__(config=final_config, params=params, logger=logger if logger else mock_dsg_logger)
        # self.logger.info(f"MockStrategy initialized with params: {self.params}")

    def forward(self, asset_features: torch.Tensor, current_positions: Optional[torch.Tensor] = None, timestamp: Optional[pd.Timestamp] = None) -> torch.Tensor: # Match BaseStrategy signature
        # self.logger.info(f"MockStrategy forward called")
        # Return a mock tensor of appropriate shape, e.g., (batch_size, 1, 1) for signals
        batch_size = asset_features.size(0) if asset_features.numel() > 0 else 0
        return torch.zeros(batch_size, 1, 1, device=asset_features.device)


    def generate_signals(self, processed_data_dict: Dict[str, pd.DataFrame], portfolio_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        # self.logger.info(f"MockStrategy generate_signals called")
        # Return an empty DataFrame or one with a 'signal' column
        if not processed_data_dict:
            return pd.DataFrame(columns=['signal'])
        # Example: use the first asset's data to generate a dummy signal
        first_asset_key = next(iter(processed_data_dict))
        data = processed_data_dict[first_asset_key]
        if data.empty:
            return pd.DataFrame(index=data.index, columns=['signal'])
        return pd.DataFrame({'signal': 0}, index=data.index)


    @classmethod
    def default_config(cls) -> StrategyConfig: # Return actual StrategyConfig or a compatible mock
        return MockStrategyConfig(name="MockStrategyDefault", default_params={'param1': 1, 'param2': 'test_default'})

    @classmethod
    def get_parameter_space(cls, optimizer_type: str = "genetic") -> Optional[Dict[str, Any]]:
        if optimizer_type == "genetic":
            return {
                'param1': {'type': 'int', 'low': 1, 'high': 100, 'step': 1}, # Added step for PyGAD
                'param2': {'type': 'categorical', 'choices': ['a', 'b', 'c']}
            }
        return None 

    @classmethod
    def get_parameter_space_for_nas(cls) -> Optional[Dict[str, Any]]:
         return {
            'layers': {'type': 'int', 'low': 1, 'high': 5},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']}
        }

# Mock strategy that also inherits from torch.nn.Module for NAS tests
class MockNASCompatibleStrategy(MockStrategy, torch.nn.Module): # type: ignore # Inherits from mocked torch.nn.Module
    def __init__(self, config: Optional[StrategyConfig] = None, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        # Call BaseStrategy init via MockStrategy's init
        MockStrategy.__init__(self, config=config, params=params, logger=logger)
        # Call mocked nn.Module init
        torch.nn.Module.__init__(self) # This will call MockModule.__init__
        # self.logger.info(f"MockNASCompatibleStrategy initialized with params: {self.params}")

    # Override forward from BaseStrategy for nn.Module compatibility if needed
    # This forward is for the nn.Module part, used by NAS.
    # The BaseStrategy.forward (for signals) is still available.
    def forward(self, x: torch.Tensor) -> torch.Tensor: # Typical nn.Module forward signature
        # self.logger.info(f"MockNASCompatibleStrategy nn.Module.forward called.")
        return x # Simple pass-through for mock

    @classmethod
    def default_config(cls) -> StrategyConfig:
        return MockStrategyConfig(name="MockNASDefault", default_params={'param_nas': 10, 'param_base': 'nas_base'})

    @classmethod
    def get_parameter_space(cls, optimizer_type: str = "genetic") -> Optional[Dict[str, Any]]:
        if optimizer_type == "nas":
            return cls.get_parameter_space_for_nas()
        # Fallback to MockStrategy's genetic space or None
        return super().get_parameter_space(optimizer_type)


class MockNASCompatibleStrategyEx(MockNASCompatibleStrategy): # For testing exceptions in NAS
    def __init__(self, config: Optional[StrategyConfig] = None, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params, logger)

class TestDynamicStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestDSG_Instance")
        self.logger.addHandler(logging.NullHandler()) # Avoid "no handler" issues

        # Define optimizer configurations that DSG will use
        self.optimizer_config_for_dsg = {
            "genetic": {"name": "GeneticOptimizer", "settings": {"population_size": 10, "num_generations": 5}},
            "nas": {"name": "NeuralArchitectureSearch", "settings": {"epochs": 1, "population_size": 5}} # Added population_size for NAS mock
        }
        self.dsg = DynamicStrategyGenerator(logger=self.logger, optimizer_config=self.optimizer_config_for_dsg)
        
        self.mock_fitness_function = MagicMock(return_value=1.0) # (params_dict, fitness_score)
        self.sample_context_df = pd.DataFrame({'close': [1,2,3,4,5]})

        # Patch GeneticOptimizer
        # Target is 'src.agent.enhanced_quantum_strategy_layer.GeneticOptimizer' because DSG imports it from there
        self.ga_patcher = patch('src.agent.enhanced_quantum_strategy_layer.GeneticOptimizer', autospec=True)
        self.MockGeneticOptimizer = self.ga_patcher.start()
        self.addCleanup(self.ga_patcher.stop)
        self.mock_ga_instance = self.MockGeneticOptimizer.return_value
        self.mock_ga_instance.run_optimizer.return_value = (None, -float('inf')) # Default GA failure (params, score)

        # Patch NeuralArchitectureSearch
        self.nas_patcher = patch('src.agent.enhanced_quantum_strategy_layer.NeuralArchitectureSearch', autospec=True)
        self.MockNASOptimizer = self.nas_patcher.start()
        self.addCleanup(self.nas_patcher.stop)
        self.mock_nas_instance = self.MockNASOptimizer.return_value
        # NAS run_optimizer might return (model_config_dict, performance_score)
        self.mock_nas_instance.run_optimizer.return_value = (None, -float('inf')) 

    def test_generate_strategy_no_optimizer_config_in_dsg(self):
        """Test DSG with no optimizer_config at all."""
        dsg_no_opt = DynamicStrategyGenerator(logger=self.logger, optimizer_config=None)
        strategy = dsg_no_opt.generate_new_strategy(MockStrategy)
        self.assertIsInstance(strategy, MockStrategy)
        self.assertEqual(strategy.params['param1'], 1) # Default from MockStrategy
        self.MockGeneticOptimizer.assert_not_called()
        self.MockNASOptimizer.assert_not_called()

    def test_generate_strategy_no_optimizer_specified_in_call(self):
        """Test DSG with optimizer_config, but optimizer_type=None in call."""
        strategy = self.dsg.generate_new_strategy(MockStrategy, optimizer_type=None)
        self.assertIsInstance(strategy, MockStrategy)
        self.assertEqual(strategy.params['param1'], 1)
        self.MockGeneticOptimizer.assert_not_called()
        self.MockNASOptimizer.assert_not_called()

    def test_generate_strategy_with_initial_parameters(self):
        initial_params = {'param1': 50, 'param_new': True}
        strategy = self.dsg.generate_new_strategy(MockStrategy, initial_parameters=initial_params, optimizer_type=None)
        self.assertEqual(strategy.params['param1'], 50)
        self.assertEqual(strategy.params['param2'], 'test_default') # From default_config
        self.assertTrue(strategy.params['param_new'])

    def test_generate_strategy_with_config_override(self):
        override_config = MockStrategyConfig(name="Overridden", default_params={'param1': 99, 'override_param': 'yes'})
        strategy = self.dsg.generate_new_strategy(MockStrategy, strategy_config_override=override_config, optimizer_type=None)
        self.assertEqual(strategy.config.name, "Overridden")
        self.assertEqual(strategy.params['param1'], 99)
        self.assertEqual(strategy.params['override_param'], 'yes')
        self.assertNotIn('param2', strategy.params)

    def test_generate_strategy_invalid_strategy_class(self):
        with patch.object(self.logger, 'error') as mock_log_error:
            strategy = self.dsg.generate_new_strategy(str, optimizer_type=None) # type: ignore
            self.assertIsNone(strategy)
            mock_log_error.assert_any_call(f"Invalid strategy_class provided: {str}. It must be a subclass of BaseStrategy.")

    def test_generate_strategy_with_genetic_optimizer(self):
        optimized_ga_params = {'param1': 77, 'param2': 'b'}
        self.mock_ga_instance.run_optimizer.return_value = (optimized_ga_params, 0.95)
        
        strategy = self.dsg.generate_new_strategy(
            MockStrategy,
            optimizer_type="genetic",
            fitness_function=self.mock_fitness_function, # For GA wrapper
            context=self.sample_context_df # For GA wrapper
        )
        self.assertIsInstance(strategy, MockStrategy)
        self.MockGeneticOptimizer.assert_called_once()
        ga_call_args = self.MockGeneticOptimizer.call_args[1] # kwargs
        self.assertIsNotNone(ga_call_args.get('fitness_function'))
        self.assertIsNotNone(ga_call_args.get('param_space'))
        self.assertEqual(ga_call_args.get('logger'), self.dsg.logger)
        self.assertEqual(ga_call_args.get('ga_settings'), self.optimizer_config_for_dsg['genetic']['settings'])
        
        self.mock_ga_instance.run_optimizer.assert_called_once()
        # run_optimizer_call_args = self.mock_ga_instance.run_optimizer.call_args[1] # kwargs
        # self.assertEqual(run_optimizer_call_args.get('current_context'), self.sample_context_df) # Context passed to wrapper

        self.assertEqual(strategy.params['param1'], 77)
        self.assertEqual(strategy.params['param2'], 'b')

    def test_generate_strategy_with_nas_optimizer(self):
        optimized_nas_params = {'layers': 3, 'activation': 'tanh'} # NAS returns architectural params
        self.mock_nas_instance.run_optimizer.return_value = (optimized_nas_params, 0.88)
        
        strategy = self.dsg.generate_new_strategy(
            MockNASCompatibleStrategy, 
            optimizer_type="nas",
            fitness_function=self.mock_fitness_function, # For NAS wrapper
            context=self.sample_context_df # For NAS wrapper
        )
        self.assertIsInstance(strategy, MockNASCompatibleStrategy)
        self.MockNASOptimizer.assert_called_once()
        nas_call_args = self.MockNASOptimizer.call_args[1] # kwargs
        self.assertIsNotNone(nas_call_args.get('fitness_function')) # Fitness func for NAS
        self.assertIsNotNone(nas_call_args.get('param_space')) # NAS param space
        self.assertEqual(nas_call_args.get('logger'), self.dsg.logger)
        self.assertEqual(nas_call_args.get('nas_settings'), self.optimizer_config_for_dsg['nas']['settings'])

        self.mock_nas_instance.run_optimizer.assert_called_once()
        # run_optimizer_call_args = self.mock_nas_instance.run_optimizer.call_args[1]
        # self.assertEqual(run_optimizer_call_args.get('current_context'), self.sample_context_df) # Context passed to wrapper

        # NAS optimized params should be in the strategy's params
        self.assertEqual(strategy.params['layers'], 3)
        self.assertEqual(strategy.params['activation'], 'tanh')
        # Check if base params from MockNASCompatibleStrategy.default_config() are also there
        self.assertEqual(strategy.params['param_nas'], 10)


    def test_generate_strategy_optimizer_fails(self):
        self.mock_ga_instance.run_optimizer.return_value = (None, -float('inf')) # Optimizer fails
        with patch.object(self.logger, 'warning') as mock_log_warning:
            strategy = self.dsg.generate_new_strategy(
                MockStrategy,
                optimizer_type="genetic",
                fitness_function=self.mock_fitness_function,
                context=self.sample_context_df
            )
            self.assertIsInstance(strategy, MockStrategy) 
            self.assertEqual(strategy.params['param1'], 1) # Default
            mock_log_warning.assert_any_call(f"Optimizer GeneticOptimizer failed to find optimal parameters for {MockStrategy.__name__}. Using default/initial parameters.")

    def test_generate_strategy_nas_optimizer_exception(self):
        self.mock_nas_instance.run_optimizer.side_effect = Exception("NAS Error")
        with patch.object(self.logger, 'error') as mock_log_error:
            strategy = self.dsg.generate_new_strategy(
                MockNASCompatibleStrategyEx,
                optimizer_type="nas",
                fitness_function=self.mock_fitness_function,
                context=self.sample_context_df
            )
            self.assertIsInstance(strategy, MockNASCompatibleStrategyEx) 
            # Check for default params from MockNASCompatibleStrategyEx -> MockNASCompatibleStrategy -> MockStrategy
            self.assertEqual(strategy.params.get('param_nas', MockNASCompatibleStrategy.default_config().default_params.get('param_nas')), 
                             MockNASCompatibleStrategy.default_config().default_params.get('param_nas'))

            mock_log_error.assert_any_call(f"Error during Neural Architecture Search for {MockNASCompatibleStrategyEx.__name__}: NAS Error", exc_info=True)


    def test_optimizer_config_override_in_call(self):
        override_opt_config = {"population_size": 100, "num_generations": 2} # GA settings
        self.mock_ga_instance.run_optimizer.return_value = (({'param1': 11}), 0.91)

        self.dsg.generate_new_strategy(
            MockStrategy,
            optimizer_type="genetic",
            optimizer_config=override_opt_config, # Override at call time
            fitness_function=self.mock_fitness_function,
            context=self.sample_context_df
        )
        ga_call_args = self.MockGeneticOptimizer.call_args[1]
        self.assertEqual(ga_call_args.get('ga_settings'), override_opt_config)

    def test_unsupported_optimizer_type(self):
        with patch.object(self.logger, 'error') as mock_log_error:
            strategy = self.dsg.generate_new_strategy(MockStrategy, optimizer_type="unknown_opt")
            self.assertIsInstance(strategy, MockStrategy) 
            self.assertEqual(strategy.params['param1'], 1)
            mock_log_error.assert_any_call("Unsupported optimizer type: unknown_opt or optimizer not configured.")
            
    def test_initial_params_and_optimizer_precedence(self):
        initial_params = {'param1': 50, 'param2': 'initial_b'}
        optimized_params = {'param1': 77, 'param_new_opt': 'optimized_val'}
        self.mock_ga_instance.run_optimizer.return_value = (optimized_params, 0.95)

        strategy = self.dsg.generate_new_strategy(
            MockStrategy,
            initial_parameters=initial_params,
            optimizer_type="genetic",
            fitness_function=self.mock_fitness_function,
            context=self.sample_context_df
        )
        # Merge logic: default -> initial -> optimized
        self.assertEqual(strategy.params['param1'], 77) # Optimizer wins
        self.assertEqual(strategy.params['param2'], 'initial_b') # From initial, as optimizer didn't provide it
        self.assertEqual(strategy.params['param_new_opt'], 'optimized_val') # From optimizer

if __name__ == '__main__':
    unittest.main()

