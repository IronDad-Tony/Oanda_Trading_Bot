import sys
import types
import copy
from abc import ABCMeta
import numpy as np
from unittest.mock import MagicMock
from typing import Dict, List, Any, Optional, Type, Tuple, Callable # Added for mock's type hints

# --- BEGIN COMPREHENSIVE TORCH MOCKING (Corrected Version) ---
# This block must run BEFORE any 'src' module that imports 'torch' is loaded.

# --- Main Mock Torch Module (Moved Up)---
mock_torch = types.ModuleType('torch')

class MockDevice: # Define MockDevice before it's used by _MOCK_TORCH_DEFAULT_CPU_DEVICE
    def __init__(self, device_type):
        self.type = device_type
    def __str__(self): return self.type
    def __repr__(self): return f"device(type='{self.type}')"

_MOCK_TORCH_DEFAULT_CPU_DEVICE = MockDevice('cpu') # Defined early

class MockTensor:
    def __init__(self, data, dtype=None, device=None):
        self.data = data
        self._is_mps = False
        self.dtype = dtype
        self._device = device if device else _MOCK_TORCH_DEFAULT_CPU_DEVICE
        self.requires_grad = False

    def to(self, device_or_dtype_or_tensor, non_blocking=False):
        target_device_obj = None
        target_dtype_obj = None

        if isinstance(device_or_dtype_or_tensor, MockDevice):
            target_device_obj = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, str):
            device_type_str = device_or_dtype_or_tensor.split(':', 1)[0]
            if device_type_str in ['cpu', 'cuda', 'mps']:
                target_device_obj = MockDevice(device_type_str)
        elif isinstance(device_or_dtype_or_tensor, types.SimpleNamespace): # types.SimpleNamespace used for dtypes
            target_dtype_obj = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, MockTensor):
            target_device_obj = device_or_dtype_or_tensor.device
            target_dtype_obj = device_or_dtype_or_tensor.dtype
        
        if target_device_obj:
            self._device = target_device_obj
        if target_dtype_obj:
            self.dtype = target_dtype_obj
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

    def mean(self, dim=None, keepdim=False):
        res = np.mean(self.data, axis=dim)
        if keepdim and dim is not None:
            res = np.expand_dims(res, axis=dim)
        return MockTensor(res if self.data else 0, dtype=self.dtype, device=self._device)

    def sum(self, dim=None, keepdim=False, dtype=None):
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
        return s[dim] if dim < len(s) else 1
    def unsqueeze(self, dim): return self # Simplified
    def squeeze(self, dim=None): return self # Simplified
    def view(self, *shape): return self # Simplified
    def permute(self, *dims): return self # Simplified
    def transpose(self, dim0, dim1): return self # Simplified
    def contiguous(self): return self
    def fill_(self, value):
        if self.data is not None: self.data = np.full_like(self.data, value).tolist()
        return self
    @property
    def device(self): return self._device
    def item(self): return self.data[0] if isinstance(self.data, list) and len(self.data)==1 and isinstance(self.data[0], (int, float, bool)) else self.data
    def argmax(self, dim=None, keepdim=False): return MockTensor(np.argmax(self.data, axis=dim), device=self._device)
    def max(self, dim=None, keepdim=False):
        if dim is None: return MockTensor(np.max(self.data), device=self._device)
        else:
            max_val = np.max(self.data, axis=dim)
            argmax_val = np.argmax(self.data, axis=dim)
            if keepdim:
                max_val = np.expand_dims(max_val, axis=dim)
                argmax_val = np.expand_dims(argmax_val, axis=dim)
            return MockTensor(max_val, device=self._device), MockTensor(argmax_val, device=self._device)
    def min(self, dim=None, keepdim=False):
        if dim is None: return MockTensor(np.min(self.data), device=self._device)
        else:
            min_val = np.min(self.data, axis=dim)
            argmin_val = np.argmin(self.data, axis=dim)
            if keepdim:
                min_val = np.expand_dims(min_val, axis=dim)
                argmin_val = np.expand_dims(argmin_val, axis=dim)
            return MockTensor(min_val, device=self._device), MockTensor(argmin_val, device=self._device)

    def backward(self, gradient=None, retain_graph=None, create_graph=False): pass # Simplified
    @property
    def grad(self):
        if not hasattr(self, '_grad'): self._grad = None # Initialize if not present
        return self._grad
    @grad.setter
    def grad(self, value): self._grad = value
    def requires_grad_(self, requires_grad=True): self.requires_grad = requires_grad; return self
    def numel(self): return np.prod(self.shape)

class MockParameter(MockTensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, device=_MOCK_TORCH_DEFAULT_CPU_DEVICE) # Ensure default device consistency
        self.requires_grad = requires_grad

class MockModule(metaclass=ABCMeta): # Use ABCMeta for abstract base class behavior
    def __init__(self, *args, **kwargs): # Allow *args, **kwargs for broader compatibility
        self._parameters: Dict[str, MockParameter] = {}
        self._modules: Dict[str, 'MockModule'] = {} # Corrected: Removed extra single quote
        self._buffers: Dict[str, MockTensor] = {}
        self._is_mock_module = True # Flag to identify mock modules
        self._is_mps = False # Default MPS status
        self._device = _MOCK_TORCH_DEFAULT_CPU_DEVICE # Default device
        self.training = True # Default training mode

    def to(self, device_or_dtype_or_tensor, non_blocking=False):
        target_device = None
        target_dtype = None

        if isinstance(device_or_dtype_or_tensor, MockDevice):
            target_device = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, str):
            device_type_str = device_or_dtype_or_tensor.split(':', 1)[0]
            if device_type_str in ['cpu', 'cuda', 'mps']:
                 target_device = MockDevice(device_type_str)
        elif isinstance(device_or_dtype_or_tensor, types.SimpleNamespace): # For dtypes
            target_dtype = device_or_dtype_or_tensor
        elif isinstance(device_or_dtype_or_tensor, MockTensor): # If a tensor is passed
            target_device = device_or_dtype_or_tensor.device
            target_dtype = device_or_dtype_or_tensor.dtype

        if target_device:
            self._device = target_device
            # Recursively move parameters, modules, and buffers
            for p_name in self._parameters: self._parameters[p_name].to(target_device)
            for m_name in self._modules: self._modules[m_name].to(target_device)
            for b_name in self._buffers: self._buffers[b_name].to(target_device)
        
        if target_dtype:
            # Recursively change dtype for parameters and buffers
            for p_name in self._parameters: self._parameters[p_name].to(target_dtype)
            # Modules typically don't have a single dtype, their sub-parameters/buffers do
            for b_name in self._buffers: self._buffers[b_name].to(target_dtype)
        return self

    def train(self, mode=True): self.training = mode; return self
    def eval(self): self.training = False; return self

    def parameters(self, recurse: bool = True) -> List[MockParameter]:
        params_list = list(self._parameters.values())
        if recurse:
            for m in self._modules.values():
                params_list.extend(m.parameters(True)) # Ensure recurse=True for submodules
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
    def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs) # Standard PyTorch behavior
    def forward(self, *args, **kwargs): raise NotImplementedError # Must be implemented by subclasses
    def apply(self, fn): # Apply a function to self and all submodules
        fn(self)
        for m in self._modules.values(): m.apply(fn)
        return self
    def state_dict(self, destination=None, prefix='', keep_vars=False): return {} # Simplified
    def load_state_dict(self, state_dict, strict=True): pass # Simplified
    def cuda(self, device=None): self.to(MockDevice('cuda')); return self # Convenience method
    def cpu(self): self.to(MockDevice('cpu')); return self # Convenience method
    def add_module(self, name: str, module: Optional['MockModule']):
        if module is None: self._modules.pop(name, None) # Remove if None
        else: self._modules[name] = module
    def register_parameter(self, name: str, param: Optional[MockParameter]):
        if param is None: self._parameters.pop(name, None) # Remove if None
        else: self._parameters[name] = param
    def register_buffer(self, name: str, tensor: Optional[MockTensor], persistent: bool = True): # persistent arg for compatibility
        if tensor is None: self._buffers.pop(name, None) # Remove if None
        else: self._buffers[name] = tensor

    # Overriding __setattr__ to automatically register MockParameter and MockModule instances
    def __setattr__(self, name, value):
        if isinstance(value, MockParameter): self.register_parameter(name, value)
        elif isinstance(value, MockModule): self._modules[name] = value
        elif isinstance(value, MockTensor): # Check if it's a buffer being set
            # This logic assumes buffers are pre-registered or handled by register_buffer
            # If a tensor is assigned to an attribute that is a registered buffer, update it.
            if name in self._buffers: self._buffers[name] = value
            else: super().__setattr__(name, value) # Regular attribute assignment
        else: super().__setattr__(name, value)

# --- Mock torch.nn ---
mock_torch_nn = types.ModuleType('torch.nn')
mock_torch_nn.Module = MockModule
mock_torch_nn.Parameter = MockParameter
# Common nn layers (can be expanded) - returning a MockModule instance for chaining/attributes
mock_torch_nn.Linear = lambda *args, **kwargs: MockModule()
mock_torch_nn.Conv1d = lambda *args, **kwargs: MockModule()
mock_torch_nn.ReLU = lambda *args, **kwargs: MockModule()
mock_torch_nn.Sigmoid = lambda *args, **kwargs: MockModule()
mock_torch_nn.Tanh = lambda *args, **kwargs: MockModule()
mock_torch_nn.Softmax = lambda *args, **kwargs: MockModule() # Takes dim argument
mock_torch_nn.Dropout = lambda *args, **kwargs: MockModule()
mock_torch_nn.BatchNorm1d = lambda *args, **kwargs: MockModule()
mock_torch_nn.LayerNorm = lambda *args, **kwargs: MockModule()
mock_torch_nn.Embedding = lambda *args, **kwargs: MockModule()
mock_torch_nn.LSTM = lambda *args, **kwargs: MockModule()
mock_torch_nn.GRU = lambda *args, **kwargs: MockModule()
mock_torch_nn.TransformerEncoderLayer = lambda *args, **kwargs: MockModule()
mock_torch_nn.TransformerEncoder = lambda *args, **kwargs: MockModule()
mock_torch_nn.Sequential = lambda *args: MockModule() # Accepts modules as args
mock_torch_nn.ModuleList = lambda modules=None: MockModule() # Accepts list of modules
mock_torch_nn.Identity = lambda *args, **kwargs: MockModule()
# Common loss functions
mock_torch_nn.CrossEntropyLoss = lambda *args, **kwargs: MockModule()
mock_torch_nn.MSELoss = lambda *args, **kwargs: MockModule()

# --- Mock torch.nn.functional ---
mock_torch_nn_functional = types.ModuleType('torch.nn.functional')
mock_torch_nn_functional.relu = lambda x, *args, **kwargs: x # Pass-through
mock_torch_nn_functional.softmax = lambda x, *args, **kwargs: x # Pass-through, dim arg
mock_torch_nn_functional.sigmoid = lambda x, *args, **kwargs: x # Pass-through
mock_torch_nn_functional.tanh = lambda x, *args, **kwargs: x # Pass-through
mock_torch_nn_functional.dropout = lambda x, *args, **kwargs: x # Pass-through
mock_torch_nn_functional.layer_norm = lambda x, *args, **kwargs: x # Pass-through
mock_torch_nn_functional.cross_entropy = lambda *args, **kwargs: MockTensor([0.0]) # Returns a scalar tensor
mock_torch_nn_functional.mse_loss = lambda *args, **kwargs: MockTensor([0.0]) # Returns a scalar tensor
mock_torch_nn_functional.gelu = lambda x, *args, **kwargs: x # GELU activation
mock_torch_nn_functional.adaptive_avg_pool2d = lambda x, *args, **kwargs: x # Adaptive pooling
mock_torch_nn_functional.gumbel_softmax = MagicMock(name="MockGumbelSoftmax", return_value=MockTensor([0.1, 0.9])) # Mock for gumbel_softmax
mock_torch_nn.functional = mock_torch_nn_functional
mock_torch.nn = mock_torch_nn

# --- Mock torch.optim ---
mock_torch_optim = types.ModuleType('torch.optim')
class MockOptimizer:
    def __init__(self, params, lr=0.001, **kwargs): # Common optimizer signature
        self.param_groups = [{'params': list(params), 'lr': lr, **kwargs}] # Store params
        self.state = {} # For optimizer state like momentum
    def zero_grad(self, set_to_none: bool = False): # PyTorch 1.7+ set_to_none
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'grad') and p.grad is not None:
                    if set_to_none: p.grad = None
                    else:
                        if hasattr(p.grad, 'detach'): p.grad = p.grad.detach() # Detach first
                        p.grad.zero_() # Then zero
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: # Closure for some optimizers
        loss = None
        if closure is not None: loss = closure()
        return loss
mock_torch_optim.Adam = MockOptimizer
mock_torch_optim.SGD = MockOptimizer
mock_torch_optim.AdamW = MockOptimizer # Common modern optimizer
mock_torch.optim = mock_torch_optim

# --- Mock torch.cuda ---
mock_torch_cuda = types.ModuleType('torch.cuda')
mock_torch_cuda.is_available = lambda: False # Mock CUDA availability
mock_torch_cuda.device_count = lambda: 0
mock_torch_cuda.current_device = lambda: -1 # Or raise error if not available
mock_torch_cuda.get_device_name = lambda device=None: ""
mock_torch.cuda = mock_torch_cuda

# --- Mock torch.backends.mps ---
mock_torch_backends = types.ModuleType('torch.backends')
mock_torch_backends_mps = types.ModuleType('torch.backends.mps')
mock_torch_backends_mps.is_available = lambda: False # Mock MPS availability
mock_torch_backends_mps.is_built = lambda: False # Mock MPS build status
mock_torch_backends.mps = mock_torch_backends_mps
mock_torch.backends = mock_torch_backends

# --- Top-level torch attributes and functions ---
mock_torch.Tensor = MockTensor
mock_torch.tensor = lambda data, dtype=None, device=None, requires_grad=False: MockTensor(data, dtype=dtype, device=device) # Factory function
mock_torch.is_tensor = lambda obj: isinstance(obj, MockTensor)
mock_torch.from_numpy = lambda ndarray: MockTensor(ndarray.tolist()) # Convert numpy to MockTensor
mock_torch.zeros = lambda *size, **kwargs: MockTensor(np.zeros(size).tolist(), dtype=kwargs.get('dtype'), device=kwargs.get('device'))
mock_torch.ones = lambda *size, **kwargs: MockTensor(np.ones(size).tolist(), dtype=kwargs.get('dtype'), device=kwargs.get('device'))
mock_torch.randn = lambda *size, **kwargs: MockTensor(np.random.randn(*size).tolist(), dtype=kwargs.get('dtype'), device=kwargs.get('device'))
mock_torch.empty = lambda *size, **kwargs: MockTensor(np.empty(size).tolist(), dtype=kwargs.get('dtype'), device=kwargs.get('device'))
mock_torch.arange = lambda *args, **kwargs: MockTensor(np.arange(*args).tolist(), dtype=kwargs.get('dtype'), device=kwargs.get('device'))
mock_torch.manual_seed = lambda seed: None # No-op
mock_torch.set_grad_enabled = lambda mode: None # No-op
mock_torch.no_grad = lambda: MagicMock() # Context manager, returns a mock
mock_torch.save = lambda obj, f, *args, **kwargs: None # No-op save
mock_torch.load = lambda f, *args, **kwargs: {} # Return empty dict or mock object for load
# Dtypes (represented as SimpleNamespace for attribute access like torch.float32)
mock_torch.float32 = types.SimpleNamespace()
mock_torch.float = mock_torch.float32 # Alias
mock_torch.int64 = types.SimpleNamespace()
mock_torch.long = mock_torch.int64 # Alias
mock_torch.bool = types.SimpleNamespace()
mock_torch.bfloat16 = types.SimpleNamespace() # For completeness
mock_torch.get_default_dtype = lambda: mock_torch.float32
mock_torch.set_default_dtype = lambda d: None # No-op
# Common math functions (can delegate to MockTensor methods or numpy)
mock_torch.sigmoid = lambda input, *args, **kwargs: input.sigmoid() if isinstance(input, MockTensor) else MockTensor(1 / (1 + np.exp(-input)))
mock_torch.tanh = lambda input, *args, **kwargs: input.tanh() if isinstance(input, MockTensor) else MockTensor(np.tanh(input))
mock_torch.exp = lambda input, *args, **kwargs: input.exp() if isinstance(input, MockTensor) else MockTensor(np.exp(input))
mock_torch.sqrt = lambda input, *args, **kwargs: input.sqrt() if isinstance(input, MockTensor) else MockTensor(np.sqrt(input))

# --- torch.device constructor ---
# Needs to handle various ways torch.device can be called (e.g. "cuda", "cuda:0", MockDevice instance)
def _mock_torch_device_constructor(device_arg=None):
    if isinstance(device_arg, str):
        dev_type = device_arg.split(':',1)[0] if ':' in device_arg else device_arg
        # Basic validation for common device types
        if dev_type in ['cpu', 'cuda', 'mps'] or "cuda" in dev_type : # "cuda:0" etc.
            return MockDevice(dev_type)
    if isinstance(device_arg, MockDevice): # If a MockDevice instance is passed
        return device_arg
    # Default to CPU if no valid argument or None
    return _MOCK_TORCH_DEFAULT_CPU_DEVICE
mock_torch.device = MagicMock(side_effect=_mock_torch_device_constructor)

# --- Sys Path Mocking ---
# This ensures that any subsequent 'import torch' or 'from torch import nn'
# will use our mock objects instead of the real PyTorch library.
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['torch.optim'] = mock_torch.optim
sys.modules['torch.cuda'] = mock_torch.cuda
sys.modules['torch.backends'] = mock_torch.backends
sys.modules['torch.backends.mps'] = mock_torch.backends.mps
# --- END COMPREHENSIVE TORCH MOCKING (Corrected Version) ---

# filepath: c:\\Users\\tonyh\\Oanda_Trading_Bot\\tests\\unit_tests\\test_risk_management_strategies.py
import pytest
import torch
import pandas as pd
import numpy as np
import logging
from src.agent.strategies.base_strategy import StrategyConfig
from src.agent.strategies.risk_management_strategies import (
    MaxDrawdownControlStrategy, # Corrected class name
    DynamicHedgingStrategy,
    RiskParityStrategy,
    VaRControlStrategy
)

# Mock logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mock_config(name: str, params: dict = None, applicable_assets: list = None) -> StrategyConfig:
    default_params = {
        'instrument_key': 'EUR_USD', # Default instrument for testing
        'feature_indices': {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4} # Example indices
    }
    if params:
        default_params.update(params)
    
    return StrategyConfig(
        name=name,
        description=f"Test config for {name}",
        default_params=default_params,
        applicable_assets=applicable_assets if applicable_assets is not None else ['EUR_USD']
    )

class TestMaximumDrawdownControlStrategy:
    def test_initialization(self):
        config = MaxDrawdownControlStrategy.default_config() # Corrected class name
        strategy = MaxDrawdownControlStrategy(config, logger=logger) # Corrected class name
        assert strategy.config.name == "MaxDrawdownControlStrategy"
        assert strategy.params['max_drawdown_limit'] == 0.10
        assert strategy.instrument_key is None # Default config has no instrument_key

        custom_params = {'max_drawdown_limit': 0.05, 'instrument_key': 'USD_JPY', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("MaxDrawdownControlStrategyCustom", params=custom_params)
        strategy_custom = MaxDrawdownControlStrategy(config_custom, params=custom_params, logger=logger) # Corrected class name
        assert strategy_custom.params['max_drawdown_limit'] == 0.05
        assert strategy_custom.instrument_key == 'USD_JPY'
        assert strategy_custom.feature_indices == {'Close': 0}

    def test_forward_no_drawdown(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, 101, 102, 103, 104 (no drawdown)
        asset_features = torch.tensor([[[100.0], [101.0], [102.0], [103.0], [104.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        
        assert signals.shape == (1, 5, 1)
        assert torch.all(signals == 0.0).item() # No risk reduction signal

    def test_forward_with_drawdown(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, 105, 90 (drawdown > 10% from 105), 92, 85 (drawdown > 10% from 105)
        # HWM:    100, 105, 105, 105, 105
        # Drawdown: 0,   0, (105-90)/105=0.14, (105-92)/105=0.12, (105-85)/105=0.19
        asset_features = torch.tensor([[[100.0], [105.0], [90.0], [92.0], [85.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        
        expected_signals = torch.tensor([[[0.0], [0.0], [-1.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_increasing_prices(self):
        params = {'max_drawdown_limit': 0.05, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        asset_features = torch.tensor([[[10.0], [11.0], [12.0], [13.0], [14.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        assert torch.all(signals == 0.0).item()

    def test_forward_with_nan_prices(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        
        # Prices: 100, NaN, 90 (HWM should be 100, drawdown (100-90)/100 = 0.1, not > 0.1)
        # HWM: 100, 100 (from last valid), 100
        # Drawdown: 0, 0 (fillna), (100-90)/100 = 0.1
        asset_features_nan = torch.tensor([[[100.0], [float('nan')], [90.0]]], dtype=torch.float32).to(DEVICE)
        signals_nan = strategy.forward(asset_features_nan)
        expected_signals_nan = torch.tensor([[[0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE) # Drawdown is 0.1, not > 0.1
        assert torch.allclose(signals_nan, expected_signals_nan)

        # Prices: 100, 105, NaN, 80 (HWM 105, drawdown (105-80)/105 = 0.23 > 0.1)
        # HWM: 100, 105, 105, 105
        # Drawdown: 0, 0, 0 (fillna), (105-80)/105 = 0.238
        asset_features_nan_2 = torch.tensor([[[100.0], [105.0], [float('nan')], [80.0]]], dtype=torch.float32).to(DEVICE)
        signals_nan_2 = strategy.forward(asset_features_nan_2)
        expected_signals_nan_2 = torch.tensor([[[0.0], [0.0], [0.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert torch.allclose(signals_nan_2, expected_signals_nan_2)

    def test_forward_empty_input(self):
        params = {'max_drawdown_limit': 0.1, 'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("MaxDrawdownControlStrategy", params=params)
        strategy = MaxDrawdownControlStrategy(config, params=params, logger=logger) # Corrected class name
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE) # seq_len is 0
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

        asset_features_empty_batch = torch.empty((0, 5, 1), dtype=torch.float32).to(DEVICE) # batch_size is 0
        signals_batch = strategy.forward(asset_features_empty_batch)
        assert signals_batch.shape == (0, 5, 1)

class TestDynamicHedgingStrategy:
    def test_initialization(self):
        config = DynamicHedgingStrategy.default_config()
        strategy = DynamicHedgingStrategy(config, logger=logger)
        assert strategy.config.name == "DynamicHedgingStrategy"
        assert strategy.params['atr_period'] == 14
        assert strategy.instrument_key is None

        custom_params = {'atr_period': 10, 'atr_multiplier_threshold': 1.5, 'instrument_key': 'GBP_USD', 
                         'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config_custom = get_mock_config("DynamicHedgingStrategyCustom", params=custom_params)
        strategy_custom = DynamicHedgingStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['atr_period'] == 10
        assert strategy_custom.params['atr_multiplier_threshold'] == 1.5
        assert strategy_custom.instrument_key == 'GBP_USD'
        assert strategy_custom.feature_indices == {'High': 0, 'Low': 1, 'Close': 2}

    def test_forward_no_hedge(self):
        params = {'atr_period': 3, 'atr_multiplier_threshold': 2.0, 'instrument_key': 'EUR_USD', 
                  'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        
        # Data: High, Low, Close
        # Price change small relative to ATR
        asset_features = torch.tensor([[
            [1.10, 1.00, 1.05],  # ATR will be NaN/0 initially
            [1.12, 1.02, 1.08],  
            [1.15, 1.05, 1.10],  # ATR calculated from here
            [1.16, 1.06, 1.11],  # Close.shift(1)-Close = 1.10-1.11 = -0.01. ATR ~0.08. Ratio small.
            [1.18, 1.08, 1.12]   # Close.shift(1)-Close = 1.11-1.12 = -0.01. Ratio small.
        ]], dtype=torch.float32).to(DEVICE)
        
        signals = strategy.forward(asset_features)
        assert signals.shape == (1, 5, 1)
        # ATR calculation needs a few periods, first signals might be 0 due to NaN ATR or small changes
        # Price change: positive if price drops (Close.shift(1) - Close)
        # df['price_change'] = df['Close'].shift(1) - df['Close']
        # signals['signal'] = np.where(processed_data['price_change_vs_atr'] > self.atr_multiplier_threshold, -1, 0)
        # Expected: 0, 0, 0, 0, 0 (assuming price_change_vs_atr does not exceed threshold)
        assert torch.all(signals == 0.0).item()


    def test_forward_hedge_triggered(self):
        params = {'atr_period': 3, 'atr_multiplier_threshold': 1.0, 'instrument_key': 'EUR_USD', 
                  'feature_indices': {'High': 0, 'Low': 1, 'Close': 2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)

        # Data: High, Low, Close
        # Large price drop relative to ATR
        asset_features = torch.tensor([[
            [1.10, 1.08, 1.09], # t0
            [1.10, 1.08, 1.09], # t1
            [1.10, 1.08, 1.09], # t2, ATR will be small, e.g. ~0.01
            [1.10, 1.00, 1.00], # t3, Price drop: 1.09 - 1.00 = 0.09. ATR ~0.03 ((0.02+0.02+0.09)/3). 0.09/0.03 = 3 > 1.0
            [1.02, 0.98, 1.01]  # t4, Price rise: 1.00 - 1.01 = -0.01. No signal
        ]], dtype=torch.float32).to(DEVICE)
        
        signals = strategy.forward(asset_features)
        # Expected signals: 0, 0, 0, -1 (hedge), 0
        # ATR for t0,t1,t2 will be small or NaN initially.
        # For t3: H=1.1, L=1.0, C=1.0. PrevC=1.09. TRs: (1.1-1.08)=0.02, (1.1-1.08)=0.02, (1.1-1.08)=0.02. ATR(3) for C[2] is ~0.02
        # Price change at t3: C[2]-C[3] = 1.09 - 1.00 = 0.09.
        # ATR at t3 (using C[0,1,2] for ATR calc affecting C[2]'s ATR, then C[0,1,2,3] for C[3]'s ATR):
        # df for ATR: H[1.1,1.1,1.1,1.1], L[1.08,1.08,1.08,1.0], C[1.09,1.09,1.09,1.0]
        # ATRs: nan, nan, 0.02, (0.02*2 + (1.1-1.0))/3 = (0.04+0.1)/3 = 0.14/3 = 0.0466
        # price_change_vs_atr for t3: 0.09 / 0.0466 = 1.93 > 1.0. So signal -1.
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'High':0,'Low':1,'Close':2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 3), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

    def test_forward_insufficient_data_for_atr(self):
        params = {'atr_period': 5, 'instrument_key': 'EUR_USD', 'feature_indices': {'High':0,'Low':1,'Close':2}}
        config = get_mock_config("DynamicHedgingStrategy", params=params)
        strategy = DynamicHedgingStrategy(config, params=params, logger=logger)
        # Only 2 data points, less than atr_period
        asset_features = torch.tensor([[[1.1, 1.0, 1.05], [1.12, 1.02, 1.08]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        assert signals.shape == (1, 2, 1)
        assert torch.all(signals == 0.0).item() # ATR will be NaN, so price_change_vs_atr will be 0

class TestRiskParityStrategy:
    def test_initialization(self):
        config = RiskParityStrategy.default_config()
        strategy = RiskParityStrategy(config, logger=logger)
        assert strategy.config.name == "RiskParityStrategy"
        assert strategy.params['vol_window'] == 20
        assert strategy.instrument_key is None

        custom_params = {'vol_window': 10, 'high_vol_threshold_pct': 0.03, 'low_vol_threshold_pct': 0.008, 
                         'instrument_key': 'AUD_USD', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("RiskParityStrategyCustom", params=custom_params)
        strategy_custom = RiskParityStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['vol_window'] == 10
        assert strategy_custom.params['high_vol_threshold_pct'] == 0.03
        assert strategy_custom.instrument_key == 'AUD_USD'

    def test_forward_low_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.005, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        # Prices with low volatility
        # Returns: NaN, 0.001, 0.001, 0.001. StdDev of returns will be small.
        asset_features = torch.tensor([[[100.0], [100.1], [100.2], [100.3], [100.4]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Volatility calculation needs vol_window=3 periods for returns, then std of those.
        # df['returns'] = [NaN, 0.000999, 0.000998, 0.000997, 0.000996]
        # df['volatility'] (window=3, min_periods=3 for returns, std dev calc):
        #   - index 0, 1: NaN (not enough data for full window of returns for std calc)
        #   - index 2: std(returns[0,1,2]) = std(NaN, 0.000999, 0.000998) -> NaN (due to initial NaN in returns)
        #   - index 2 (corrected logic): std(returns[0..2]) -> df['volatility'][2] is based on returns[0], returns[1], returns[2]. Since returns[0] is NaN, vol[2] is NaN.
        #   - index 3: std(returns[1,2,3]) = std(0.000999, 0.000998, 0.000997) approx 0.000001. This is < 0.005. Signal = 1.
        #   - index 4: std(returns[2,3,4]) = std(0.000998, 0.000997, 0.000996) approx 0.000001. This is < 0.005. Signal = 1.
        # Expected signals: 0 (for NaN vol), 0 (for NaN vol), 0 (for NaN vol), 1, 1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [1.0], [1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_high_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.005, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        asset_features = torch.tensor([[[100.0], [105.0], [100.0], [105.0], [100.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # df['returns'] = [NaN, 0.05, -0.0476, 0.05, -0.0476]
        # df['volatility'] (window=3, min_periods=3):
        #   - index 0, 1, 2: NaN
        #   - index 3: std(returns[1,2,3]) = std(0.05, -0.0476, 0.05) approx 0.056. This is > 0.02. Signal = -1.
        #   - index 4: std(returns[2,3,4]) = std(-0.0476, 0.05, -0.0476) approx 0.056. This is > 0.02. Signal = -1.
        # Expected: 0, 0, 0, -1, -1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_neutral_volatility(self):
        params = {'vol_window': 3, 'high_vol_threshold_pct': 0.02, 'low_vol_threshold_pct': 0.01, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        
        # Prices with volatility between low and high thresholds
        # Returns: NaN, 0.015, 0.0147, 0.0145
        asset_features = torch.tensor([[[100.0], [101.5], [103.0], [104.5], [106.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Returns: nan, 0.015, 0.01477, 0.01456, 0.01435
        # Rolling std(3): nan, nan, std([0.015, 0.01477, 0.01456]) ~0.0002. This is < low_vol_threshold (0.01)
        # Let's adjust prices for neutral:
        # Prices: 100, 101, 102, 100, 101. Returns: nan, 0.01, 0.0099, -0.0196, 0.01
        # std([0.01, 0.0099, -0.0196]) ~ 0.017. This is between 0.01 and 0.02.
        asset_features_neutral = torch.tensor([[[100.0], [101.0], [102.0], [100.0], [101.0]]], dtype=torch.float32).to(DEVICE)
        signals_neutral = strategy.forward(asset_features_neutral)
        expected_signals_neutral = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals_neutral.shape == (1, 5, 1)
        assert torch.allclose(signals_neutral, expected_signals_neutral)


    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'Close':0}}
        config = get_mock_config("RiskParityStrategy", params=params)
        strategy = RiskParityStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

class TestVaRControlStrategy:
    def test_initialization(self):
        config = VaRControlStrategy.default_config()
        strategy = VaRControlStrategy(config, logger=logger)
        assert strategy.config.name == "VaRControlStrategy"
        assert strategy.params['var_window'] == 20
        assert strategy.params['var_limit'] == 0.02
        assert strategy.instrument_key is None
        assert strategy.z_score == pytest.approx(2.3263, abs=1e-4) # for 0.99 confidence

        custom_params = {'var_window': 10, 'var_limit': 0.03, 'var_confidence': 0.95, 
                         'instrument_key': 'USD_CAD', 'feature_indices': {'Close': 0}}
        config_custom = get_mock_config("VaRControlStrategyCustom", params=custom_params)
        strategy_custom = VaRControlStrategy(config_custom, params=custom_params, logger=logger)
        assert strategy_custom.params['var_window'] == 10
        assert strategy_custom.params['var_limit'] == 0.03
        assert strategy_custom.instrument_key == 'USD_CAD'
        assert strategy_custom.z_score == pytest.approx(1.64485, abs=1e-4) # for 0.95 confidence

    def test_forward_var_below_limit(self):
        params = {'var_window': 3, 'var_limit': 0.05, 'var_confidence': 0.99, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        z_score = strategy.z_score # approx 2.3263
        
        # Prices with low volatility, so VaR should be low
        # Returns: NaN, 0.001, 0.001, 0.001
        asset_features = torch.tensor([[[100.0], [100.1], [100.2], [100.3], [100.4]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # Returns: nan, 0.000999, 0.000998, 0.000997, 0.000996
        # Rolling std(3): nan, nan, std([0.000999, 0.000998, 0.000997]) ~0.000001
        # Estimated VaR = 0.000001 * 2.3263 = 0.0000023 < 0.05
        # Expected: 0, 0, 0, 0, 0
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_var_exceeds_limit(self):
        params = {'var_window': 3, 'var_limit': 0.05, 'var_confidence': 0.99, 
                  'instrument_key': 'EUR_USD', 'feature_indices': {'Close': 0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        z_score = strategy.z_score # approx 2.3263

        asset_features = torch.tensor([[[100.0], [103.0], [100.0], [103.0], [100.0]]], dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features)
        # df['returns'] = [NaN, 0.03, -0.02912, 0.03, -0.02912]
        # df['rolling_std'] (window=3, min_periods=3):
        #   - index 0, 1, 2: NaN
        #   - index 3: std(returns[1,2,3]) = std(0.03, -0.02912, 0.03) approx 0.034.
        #     Estimated VaR = 0.034 * 2.3263 = 0.079 > 0.05. Signal = -1.
        #   - index 4: std(returns[2,3,4]) = std(-0.02912, 0.03, -0.02912) approx 0.034.
        #     Estimated VaR = 0.034 * 2.3263 = 0.079 > 0.05. Signal = -1.
        # Expected: 0, 0, 0, -1, -1
        expected_signals = torch.tensor([[[0.0], [0.0], [0.0], [-1.0], [-1.0]]], dtype=torch.float32).to(DEVICE)
        assert signals.shape == (1, 5, 1)
        assert torch.allclose(signals, expected_signals)

    def test_forward_empty_input(self):
        params = {'instrument_key': 'EUR_USD', 'feature_indices': {'Close':0}}
        config = get_mock_config("VaRControlStrategy", params=params)
        strategy = VaRControlStrategy(config, params=params, logger=logger)
        asset_features_empty = torch.empty((1, 0, 1), dtype=torch.float32).to(DEVICE)
        signals = strategy.forward(asset_features_empty)
        assert signals.shape == (1, 0, 1)

