# tests/unit_tests/test_quantum_strategies.py
# Initial imports (unified)
import sys
import types
import copy
from abc import ABCMeta
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Type, Tuple, Callable
import pytest
import os
import json
import tempfile

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
mock_torch.backends = mock_torch_backends
mock_torch.backends.mps = mock_torch_backends_mps

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
sys.modules['torch.nn'] = mock_torch_nn # Corrected: use mock_torch_nn
sys.modules['torch.nn.functional'] = mock_torch_nn_functional # Corrected: use mock_torch_nn_functional
sys.modules['torch.optim'] = mock_torch_optim # Corrected: use mock_torch_optim
sys.modules['torch.cuda'] = mock_torch_cuda # Corrected: use mock_torch_cuda
sys.modules['torch.backends'] = mock_torch_backends # Corrected: use mock_torch_backends
sys.modules['torch.backends.mps'] = mock_torch_backends_mps # Corrected: use mock_torch_backends_mps

# --- END COMPREHENSIVE TORCH MOCKING (Corrected Version) ---

import torch # This will now import the mock_torch object

# Define project_root globally
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Attempt to import from src
try:
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition, DynamicStrategyGenerator
    from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig # Ensure StrategyConfig is imported
    from src.common.config import DEVICE
    from src.agent.strategies import STRATEGY_REGISTRY
    from src.agent.strategies.trend_strategies import MomentumStrategy, MeanReversionStrategy as TrendMeanReversionStrategy, ReversalStrategy, BreakoutStrategy, TrendFollowingStrategy # For checking params and testing
    from src.agent.strategies.statistical_arbitrage_strategies import (
        MeanReversionStrategy,
        CointegrationStrategy,
        PairsTradeStrategy,
        StatisticalArbitrageStrategy,
        VolatilityBreakoutStrategy
    )
except ImportError:
    # Fallback for environments where src is not directly in PYTHONPATH
    import sys
    # 'os' is already imported globally
    # 'project_root' is already defined globally
    # Add project root to sys.path to allow src.module imports
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now try importing with src prefix
    from src.agent.enhanced_quantum_strategy_layer import EnhancedStrategySuperposition, DynamicStrategyGenerator
    from src.agent.strategies.base_strategy import BaseStrategy, StrategyConfig
    from src.common.config import DEVICE
    from src.agent.strategies import STRATEGY_REGISTRY
    from src.agent.strategies.trend_strategies import MomentumStrategy, MeanReversionStrategy as TrendMeanReversionStrategy, ReversalStrategy, BreakoutStrategy, TrendFollowingStrategy
    from src.agent.strategies.statistical_arbitrage_strategies import (
        MeanReversionStrategy,
        CointegrationStrategy,
        PairsTradeStrategy,
        StatisticalArbitrageStrategy,
        VolatilityBreakoutStrategy
    )

# Configure a logger for tests
test_logger = logging.getLogger("TestQuantumStrategies")
test_logger.setLevel(logging.DEBUG) # Or logging.INFO
# Ensure a handler is configured to see output during tests
if not test_logger.handlers:
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

    log_file_path = os.path.join(project_root, "test_quantum_strategies.log")
    file_handler = logging.FileHandler(log_file_path, mode='w') 
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    test_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(log_formatter)
    test_logger.addHandler(console_handler)

# Define LearnableMockStrategy for gradient testing
class LearnableMockStrategy(BaseStrategy):
    # MODIFIED: Align __init__ with BaseStrategy
    def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, params, logger)
        # MODIFIED: Initialize nn.Parameter without explicit device. Device placement handled by parent module.
        self.learnable_param = torch.nn.Parameter(torch.randn(1)) # Uses mocked torch.nn.Parameter

    @staticmethod
    def default_config() -> StrategyConfig: # Return StrategyConfig object
        return StrategyConfig(
            name='LearnableMockStrategy',
            description='A mock strategy with a learnable parameter for gradient testing.',
            default_params={'mock_specific_param': 100}
        )

    def forward(self,
                asset_features: torch.Tensor, # MODIFIED: Renamed and updated shape comment
                                                 # (batch_size, sequence_length, input_dim)
                current_positions: Optional[torch.Tensor] = None, # MODIFIED: Renamed and updated shape comment
                                                                    # (batch_size, 1)
                timestamp: Optional[pd.Timestamp] = None
                ) -> torch.Tensor: # Expected output: (batch_size, 1, 1) signal tensor

        # MODIFIED: asset_features_for_strategy -> asset_features
        if asset_features.numel() == 0:
            # Ensure batch_size dimension is preserved if asset_features.size(0) is valid
            batch_s = asset_features.size(0) if asset_features.dim() > 0 else 0 # Handle 0-dim tensor from numel=0
            return torch.zeros((batch_s, 1, 1), device=asset_features.device) # Uses mocked torch.zeros

        # Assuming we use the first feature (index 0) from input_dim for calculation
        # asset_features shape: (batch_size, sequence_length, input_dim)
        # MODIFIED: Corrected indexing for 3D tensor
        mean_feature = asset_features[:, :, 0].mean(dim=1, keepdim=True) # Result: (batch_size, 1)
        signal = torch.tanh(mean_feature * self.learnable_param) # Result: (batch_size, 1)

        # Output expected by EnhancedStrategySuperposition is (batch_size, 1, 1)
        return signal.unsqueeze(1) # This was correct: (batch_size, 1) -> (batch_size, 1, 1)

    # ADDED: Implement abstract method generate_signals
    def generate_signals(self, 
                         processed_data_dict: Dict[str, pd.DataFrame], 
                         portfolio_context: Optional[Dict[str, Any]] = None
                         ) -> pd.DataFrame:
        self.logger.info(f"LearnableMockStrategy.generate_signals called. Input keys: {list(processed_data_dict.keys())}")
        first_asset_data = next(iter(processed_data_dict.values()), None)
        if first_asset_data is not None and not first_asset_data.empty:
            last_timestamp = first_asset_data.index[-1]
            signal_value = 0 
            if 'feature1' in first_asset_data.columns:
                signal_value = 1 if first_asset_data['feature1'].iloc[-1] > 0 else -1
            return pd.DataFrame({'signal': [signal_value]}, index=[last_timestamp])
        else:
            return pd.DataFrame(columns=['signal'], index=pd.DatetimeIndex([]))

# Configure pytest fixtures
@pytest.fixture(scope="session")
def device():
    """Provides the torch device (CPU or CUDA) for tests."""
    return DEVICE # This is from src.common.config

@pytest.fixture
def mock_logger():
    return test_logger

@pytest.fixture
def base_config(tmp_path):
    default_strategy_params = {
        "feature_dim": 5, 
        "num_assets": 2,
        "sequence_length": 60,
        "market_feature_dim": 10,
        "adaptive_weight_config": {
            "method": "attention",
            "attention_dim": 64
        },
    }
    strategy_specific_params_overrides = {
        "MomentumStrategy": {"window": 7},
        "QuantitativeStrategy": {"expression": "close > sma(close, 20)"},
        "CointegrationStrategy": {"asset_pair": ["EUR_USD", "USD_JPY"]},
        "PairsTradeStrategy": {"asset_pair": ["EUR_USD", "USD_JPY"]},
        "AlgorithmicStrategy": {
            "rule_buy_condition": "close > placeholder_indicator(period=10)",
            "rule_sell_condition": "close < placeholder_indicator(period=10)",
            "asset_list": ["EUR_USD"]
        },
        "LearnableMockStrategy": {"mock_specific_param": 150}
    }
    return StrategyConfig(
        name="BaseTestConfig",
        description="Base configuration for testing EnhancedStrategySuperposition.",
        default_params=default_strategy_params,
        strategy_specific_params=strategy_specific_params_overrides,
        applicable_assets=["EUR_USD", "USD_JPY"]
    )

@pytest.fixture
def mock_input_features_for_real_strategies(base_config, device): # device fixture is the real one
    batch_size = 2
    num_assets = base_config.default_params["num_assets"]
    sequence_length = base_config.default_params["sequence_length"]
    feature_dim = base_config.default_params["feature_dim"] 
    market_feature_dim = base_config.default_params["market_feature_dim"]

    # Uses mocked torch.randn, but .to(device) where device is real
    asset_features_batch = torch.randn(batch_size, num_assets, sequence_length, feature_dim).to(device)
    market_state_features = torch.randn(batch_size, market_feature_dim).to(device)
    timestamps = [pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=i) for i in range(batch_size)]
    current_positions_batch = torch.randn(batch_size, num_assets, 1).to(device)
    
    return asset_features_batch, market_state_features, timestamps, current_positions_batch

# Helper to create a mock EnhancedStrategySuperposition for tests that need it
def create_mock_ess_layer(input_dim, num_strategies_to_init, overall_config, strategy_classes, logger_instance, layer_settings_override=None):
    default_layer_settings = {
        "adaptive_weight_config": overall_config.default_params.get("adaptive_weight_config"),
        "dsg_optimizer_config": None,
        "max_generated_strategies": 0,
        "innovation_engine_active": False,
        "dropout_rate": 0.1,
        "initial_temperature": 1.0,
        "use_gumbel_softmax": True,
        "strategy_input_dim": overall_config.default_params.get("feature_dim", 64)
    }
    if layer_settings_override:
        default_layer_settings.update(layer_settings_override)

    return EnhancedStrategySuperposition( # This will use mocked torch.nn.Module etc.
        input_dim=input_dim,
        num_strategies=num_strategies_to_init, 
        strategy_configs=[sc.default_config() for sc in strategy_classes if hasattr(sc, 'default_config')] + [overall_config],
        explicit_strategies=strategy_classes,
        dropout_rate=default_layer_settings["dropout_rate"],
        initial_temperature=default_layer_settings["initial_temperature"],
        use_gumbel_softmax=default_layer_settings["use_gumbel_softmax"],
        strategy_input_dim=default_layer_settings["strategy_input_dim"]
    )

class BaseStrategyTest:
    batch_size = 2
    seq_len = 20
    feature_dim = 5
    num_features_per_asset = 3

    @pytest.fixture
    def mock_logger_fixture(self):
        logger = logging.getLogger("TestQuantumStrategies")
        logger.setLevel(logging.DEBUG)
        return logger

def test_initialization_with_real_strategies(base_config: StrategyConfig, mock_logger):
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty or not imported. Skipping test with real strategies.")

    input_dim = base_config.default_params.get("feature_dim", 128)
    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim,
            num_strategies=num_strategies_to_initialize,
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=input_dim
        )
    
    assert layer.num_actual_strategies > 0
    assert layer.num_actual_strategies <= len(strategy_classes_to_test)
    assert len(layer.strategy_names) == layer.num_actual_strategies
    assert len(layer.strategies) == layer.num_actual_strategies
    for strategy_module in layer.strategies:
        assert isinstance(strategy_module, BaseStrategy)

def test_forward_with_internal_weights_real_strategies(base_config: StrategyConfig, mock_logger, mock_input_features_for_real_strategies):
    if not STRATEGY_REGISTRY:
        pytest.skip("STRATEGY_REGISTRY is empty. Skipping forward test with real strategies.")
    
    input_dim_for_attention = base_config.default_params.get("market_feature_dim", 128)
    strategy_input_feature_dim = base_config.default_params.get("feature_dim", 64)

    strategy_classes_to_test = list(STRATEGY_REGISTRY.values())
    num_strategies_to_initialize = len(strategy_classes_to_test)

    # Get the actual device from the fixture for .to(device) calls
    # The mock_input_features_for_real_strategies already puts tensors on this device
    actual_device_for_layer = mock_input_features_for_real_strategies[0].device 

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=input_dim_for_attention,
            num_strategies=num_strategies_to_initialize,
            strategy_configs=[cls.default_config() for cls in strategy_classes_to_test if hasattr(cls, 'default_config')],
            explicit_strategies=strategy_classes_to_test,
            strategy_input_dim=strategy_input_feature_dim
        )
        # layer.to(actual_device_for_layer) # Layer and its mock params will be moved to this mock device

        if layer.num_actual_strategies == 0:
            pytest.skip("No strategies loaded in ESS, cannot perform forward pass test.")

    asset_features_batch, market_state_features, timestamps, current_positions_batch = mock_input_features_for_real_strategies
    
    output_actions = layer.forward(
        asset_features_batch, 
        market_state_features=market_state_features,
        current_positions_batch=current_positions_batch,
        timestamps=timestamps
    )
    
    assert output_actions is not None
    expected_num_assets = asset_features_batch.shape[1]
    batch_size = asset_features_batch.shape[0]
    assert output_actions.shape == (batch_size, expected_num_assets, 1)


@pytest.mark.parametrize("use_gumbel_softmax_param", [True, False])
def test_strategy_combination_weights_and_execution_order(
    base_config: StrategyConfig, 
    mock_logger, 
    device, # This is the real device from fixture
    use_gumbel_softmax_param: bool
):
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    strategy_feature_input_dim = 5 
    base_config.default_params["feature_dim"] = strategy_feature_input_dim

    batch_size = 2
    num_test_assets = 2
    seq_len = 10

    mock_strategy_configs = []
    mock_strategy_classes = []

    for i in range(3):
        strat_id_numeric = i + 1
        class MockStrat(BaseStrategy):
            def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
                super().__init__(config, params, logger)
                self.numeric_id = int(self.config.name.replace("MockStrat", ""))
                self.forward_call_args = []

            @staticmethod
            def default_config(): 
                return StrategyConfig(
                    name="PLACEHOLDER_NAME", 
                    default_params={'input_dim': strategy_feature_input_dim}
                )

            def forward(self, asset_features, current_positions=None, timestamp=None):
                self.forward_call_args.append({
                    'asset_features_shape': asset_features.shape,
                    'current_positions_shape': current_positions.shape if current_positions is not None else None,
                    'timestamp': timestamp
                })
                return torch.full((asset_features.shape[0], 1, 1), float(self.numeric_id), device=asset_features.device) # Mocked torch.full

        MockStrat.__name__ = f"MockStrat{strat_id_numeric}"
        config = StrategyConfig(
            name=f"MockStrat{strat_id_numeric}", 
            default_params={'input_dim': strategy_feature_input_dim},
            description=f"Mock strategy {strat_id_numeric}"
        )
        config.input_dim = strategy_feature_input_dim 
        mock_strategy_configs.append(config)
        mock_strategy_classes.append(MockStrat)

    temp_strategy_registry = STRATEGY_REGISTRY.copy()
    for idx, strat_class in enumerate(mock_strategy_classes):
        temp_strategy_registry[mock_strategy_configs[idx].name] = strat_class

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger), \
         patch('src.agent.enhanced_quantum_strategy_layer.STRATEGY_REGISTRY', temp_strategy_registry):        
        layer = EnhancedStrategySuperposition(
            input_dim=layer_attention_input_dim,
            num_strategies=len(mock_strategy_classes),
            strategy_configs=mock_strategy_configs, 
            explicit_strategies=None, 
            strategy_input_dim=strategy_feature_input_dim,
            use_gumbel_softmax=use_gumbel_softmax_param 
        )
    # layer.to(device) # Move layer to the real device; its mock tensors will adapt
    layer.eval()

    assert layer.num_actual_strategies == len(mock_strategy_classes)
    mock_strategy_instances = list(layer.strategies)
    assert len(mock_strategy_instances) == len(mock_strategy_classes)

    asset_features_batch = torch.randn(batch_size, num_test_assets, seq_len, strategy_feature_input_dim).to(device) # Mocked randn, real device
    market_state_features = torch.randn(batch_size, layer_attention_input_dim).to(device)
    current_positions_batch = torch.randn(batch_size, num_test_assets, 1).to(device)
    timestamps = [pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=k) for k in range(batch_size)]

    mock_logger.info("Testing with External Weights")
    external_weights_input = torch.tensor([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]], dtype=torch.float).to(device) # Mocked tensor, real device
    assert external_weights_input.shape == (batch_size, layer.num_actual_strategies)

    for mock_strat_instance in mock_strategy_instances:
        mock_strat_instance.forward_call_args = []

    output_actions_ext = layer.forward(
        asset_features_batch,
        market_state_features=market_state_features, 
        current_positions_batch=current_positions_batch,
        timestamps=timestamps,
        external_weights=external_weights_input
    )

    assert output_actions_ext.shape == (batch_size, num_test_assets, 1)

    for mock_strat_instance in mock_strategy_instances:
        assert len(mock_strat_instance.forward_call_args) == num_test_assets
        for k in range(num_test_assets):
            call_arg = mock_strat_instance.forward_call_args[k]
            assert call_arg['asset_features_shape'] == (batch_size, seq_len, strategy_feature_input_dim)
            assert call_arg['current_positions_shape'] == (batch_size, 1)
            assert call_arg['timestamp'] == timestamps[0] 

    # Use mocked F.softmax
    processed_external_weights = torch.nn.functional.softmax(external_weights_input / layer.temperature.item(), dim=1)
    
    expected_output_ext = torch.zeros(batch_size, num_test_assets, 1).to(device) # Mocked zeros, real device
    for b in range(batch_size):
        for a in range(num_test_assets):
            weighted_sum = torch.tensor(0.0).to(device) # Mocked tensor for sum
            for s_idx, mock_strat_instance in enumerate(mock_strategy_instances):
                strategy_raw_output = torch.tensor(float(mock_strat_instance.numeric_id)).to(device)
                weighted_sum += processed_external_weights[b, s_idx] * strategy_raw_output
            expected_output_ext[b, a, 0] = weighted_sum
    
    assert torch.allclose(output_actions_ext, expected_output_ext, atol=1e-5) # Mocked allclose

    mock_logger.info(f"Testing with Internal Weights (Gumbel Softmax: {use_gumbel_softmax_param})")
    for mock_strat_instance in mock_strategy_instances:
        mock_strat_instance.forward_call_args = []
    
    internal_mock_logits = torch.tensor([[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]], dtype=torch.float).to(device)
    with patch.object(layer.attention_network, 'forward', MagicMock(return_value=internal_mock_logits)) as mock_attention_forward:
        output_actions_internal = layer.forward(
            asset_features_batch,
            market_state_features=market_state_features, 
            current_positions_batch=current_positions_batch,
            timestamps=timestamps,
            external_weights=None 
        )
        mock_attention_forward.assert_called_once_with(market_state_features)

    assert output_actions_internal.shape == (batch_size, num_test_assets, 1)

    for mock_strat_instance in mock_strategy_instances:
        assert len(mock_strat_instance.forward_call_args) == num_test_assets 

    current_temp = layer.temperature.item()
    expected_internal_weights = torch.nn.functional.softmax(internal_mock_logits / current_temp, dim=1) # Mocked softmax
    layer.eval() 

    expected_output_int = torch.zeros(batch_size, num_test_assets, 1).to(device) # Mocked zeros
    for b in range(batch_size):
        for a in range(num_test_assets):
            weighted_sum = torch.tensor(0.0).to(device) # Mocked tensor
            for s_idx, mock_strat_instance in enumerate(mock_strategy_instances):
                strategy_raw_output = torch.tensor(float(mock_strat_instance.numeric_id)).to(device)
                weighted_sum += expected_internal_weights[b, s_idx] * strategy_raw_output
            expected_output_int[b, a, 0] = weighted_sum
            
    assert torch.allclose(output_actions_internal, expected_output_int, atol=1e-4) # Mocked allclose
    mock_logger.info(f"Test passed with Gumbel Softmax: {use_gumbel_softmax_param}")

def test_dynamic_strategy_generator_generates_strategy(mock_logger):
    pass # Placeholder

class TestAdaptiveWeighting(unittest.TestCase):
    def setUp(self):
        self.mock_logger = logging.getLogger("TestAdaptiveWeighting")
        self.mock_logger.setLevel(logging.DEBUG)
        if not self.mock_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.mock_logger.addHandler(handler)

        self.input_dim = 10
        self.num_strategies = 3
        self.strategy_input_dim = 5
        self.batch_size = 2
        self.num_assets = 1
        self.seq_len = 7
        self.actual_device = 'cpu' # Or use DEVICE from src.common.config if available and appropriate

        self.mock_strategy_configs = []
        for i in range(self.num_strategies):
            config = StrategyConfig(
                name=f"MockAdaptiveStrategy{i+1}",
                default_params={'input_dim': self.strategy_input_dim},
                description=f"Mock adaptive strategy {i+1}"
            )
            config.input_dim = self.strategy_input_dim
            self.mock_strategy_configs.append(config)

        self.mock_strategy_classes = []
        for i in range(self.num_strategies):
            class MockStrat(BaseStrategy):
                strat_idx_capture = i 
                def __init__(self, config: StrategyConfig, params: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
                    super().__init__(config, params, logger)
                    self.numeric_id = int(self.config.name.replace("MockAdaptiveStrategy", ""))

                @staticmethod
                def default_config():
                    # This is tricky; name should ideally match. Using a fixed pattern for test.
                    # The strat_idx_capture is not directly accessible in staticmethod in a loop.
                    # For the test, the config objects passed to ESS will have the correct names.
                    return StrategyConfig(name=f"MockAdaptiveStrategy{1}", default_params={'input_dim': 5})


                def forward(self, asset_features, current_positions=None, timestamp=None):
                    return torch.full((asset_features.shape[0], 1, 1), float(self.numeric_id), device=asset_features.device) # Mocked torch

            MockStrat.__name__ = f"MockAdaptiveStrategy{i+1}" 
            self.mock_strategy_classes.append(MockStrat)
        
        self.patcher = patch('src.agent.enhanced_quantum_strategy_layer.STRATEGY_REGISTRY', STRATEGY_REGISTRY.copy())
        self.mock_registry = self.patcher.start()
        for i, strat_class in enumerate(self.mock_strategy_classes):
            self.mock_registry[self.mock_strategy_configs[i].name] = strat_class


    def tearDown(self):
        self.patcher.stop()

    def _create_layer(self, adaptive_learning_rate=0.01, performance_ema_alpha=0.1, initial_adaptive_bias=None):
        with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=self.mock_logger):
            layer = EnhancedStrategySuperposition(
                input_dim=self.input_dim,
                num_strategies=self.num_strategies,
                strategy_configs=self.mock_strategy_configs,
                strategy_input_dim=self.strategy_input_dim,
                adaptive_learning_rate=adaptive_learning_rate,
                performance_ema_alpha=performance_ema_alpha
            )
            # layer.to(self.actual_device) # Mocked .to()
            if initial_adaptive_bias is not None and layer.adaptive_bias_weights is not None:
                layer.adaptive_bias_weights.data = torch.tensor(initial_adaptive_bias, dtype=torch.float).data # Mocked tensor
            layer.eval() 
            return layer

    def test_update_adaptive_weights_basic(self):
        self.mock_logger.info("Testing test_update_adaptive_weights_basic")
        layer = self._create_layer(adaptive_learning_rate=0.1, performance_ema_alpha=0.5)
        
        initial_ema = layer.strategy_performance_ema.clone().cpu().numpy() # Mocked clone, cpu, numpy
        initial_bias = layer.adaptive_bias_weights.clone().cpu().numpy()

        rewards1 = torch.tensor([0.1, -0.05, 0.2]) # Mocked tensor
        layer.update_adaptive_weights(rewards1)
        
        expected_ema1 = (1 - 0.5) * initial_ema + 0.5 * rewards1.cpu().numpy()
        np.testing.assert_array_almost_equal(layer.strategy_performance_ema.cpu().numpy(), expected_ema1, decimal=6)

        perf_dev1 = expected_ema1 - expected_ema1.mean()
        expected_bias1 = initial_bias + 0.1 * perf_dev1
        np.testing.assert_array_almost_equal(layer.adaptive_bias_weights.cpu().numpy(), expected_bias1, decimal=6)

        rewards2 = torch.tensor([-0.1, 0.15, 0.05]) # Mocked tensor
        layer.update_adaptive_weights(rewards2)

        expected_ema2 = (1 - 0.5) * expected_ema1 + 0.5 * rewards2.cpu().numpy()
        np.testing.assert_array_almost_equal(layer.strategy_performance_ema.cpu().numpy(), expected_ema2, decimal=6)
        
        perf_dev2 = expected_ema2 - expected_ema2.mean()
        expected_bias2 = expected_bias1 + 0.1 * perf_dev2
        np.testing.assert_array_almost_equal(layer.adaptive_bias_weights.cpu().numpy(), expected_bias2, decimal=6)
        self.mock_logger.info("Finished test_update_adaptive_weights_basic")

    def test_update_adaptive_weights_edge_cases(self):
        self.mock_logger.info("Testing test_update_adaptive_weights_edge_cases")
        layer = self._create_layer()
        
        with patch.object(layer.logger, 'error') as mock_log_error:
            rewards_wrong_shape = torch.randn(self.num_strategies + 1) # Mocked randn
            layer.update_adaptive_weights(rewards_wrong_shape)
            mock_log_error.assert_called_once()
            self.assertTrue("shape mismatch" in mock_log_error.call_args[0][0])

        with patch.object(layer.logger, 'error') as mock_log_error:
            rewards_wrong_type = [0.1, 0.2, 0.3] 
            layer.update_adaptive_weights(rewards_wrong_type) # type: ignore
            mock_log_error.assert_called_once()
            self.assertTrue("must be a torch.Tensor" in mock_log_error.call_args[0][0])
        
        with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=self.mock_logger):
            empty_layer = EnhancedStrategySuperposition(input_dim=10, num_strategies=0)
        with patch.object(empty_layer.logger, 'warning') as mock_log_warning:
            empty_layer.update_adaptive_weights(torch.tensor([])) # Mocked tensor
            mock_log_warning.assert_called()
            self.assertTrue("Adaptive components not initialized or no strategies" in mock_log_warning.call_args[0][0])
        self.mock_logger.info("Finished test_update_adaptive_weights_edge_cases")


    def test_forward_with_adaptive_weights_only(self):
        self.mock_logger.info("Testing test_forward_with_adaptive_weights_only")
        initial_bias = [0.5, -0.2, 0.8]
        layer = self._create_layer(initial_adaptive_bias=initial_bias)
        layer.attention_network = None 

        asset_features = torch.randn(self.batch_size, self.num_assets, self.seq_len, self.strategy_input_dim) # Mocked randn
        
        layer.dropout = torch.nn.Identity() # Mocked Identity

        output_actions = layer.forward(asset_features)
        
        expected_logits = torch.tensor(initial_bias, dtype=torch.float).unsqueeze(0).expand(self.batch_size, -1) # Mocked tensor
        expected_strategy_weights = torch.nn.functional.softmax(expected_logits / layer.temperature.item(), dim=1) # Mocked softmax
        
        strategy_outputs = torch.tensor([[[1.0]], [[2.0]], [[3.0]]], dtype=torch.float) # Mocked tensor
        strategy_outputs_batched = strategy_outputs.squeeze().unsqueeze(0).expand(self.batch_size, -1)

        expected_combined_signal_flat = torch.sum(expected_strategy_weights * strategy_outputs_batched, dim=1) # Mocked sum
        expected_final_actions = expected_combined_signal_flat.unsqueeze(-1).unsqueeze(-1)

        self.assertTrue(torch.allclose(output_actions, expected_final_actions, atol=1e-5)) # Mocked allclose
        self.mock_logger.info("Finished test_forward_with_adaptive_weights_only")

    def test_forward_with_adaptive_and_attention_weights(self):
        self.mock_logger.info("Testing test_forward_with_adaptive_and_attention_weights")
        initial_bias = [0.1, 0.2, 0.3]
        layer = self._create_layer(initial_adaptive_bias=initial_bias)
        
        asset_features = torch.randn(self.batch_size, self.num_assets, self.seq_len, self.strategy_input_dim) # Mocked randn
        market_state_features = torch.randn(self.batch_size, self.input_dim) # Mocked randn

        layer.dropout = torch.nn.Identity() # Mocked Identity

        mock_attention_logits = torch.tensor([[0.5, 0.3, 0.2], [-0.1, 0.6, 0.4]], dtype=torch.float) # Mocked tensor
        
        with patch.object(layer.attention_network, 'forward', return_value=mock_attention_logits) as mock_attn_forward:
            output_actions = layer.forward(asset_features, market_state_features=market_state_features)
            mock_attn_forward.assert_called_once_with(market_state_features)

        adaptive_bias_expanded = torch.tensor(initial_bias, dtype=torch.float).unsqueeze(0).expand(self.batch_size, -1) # Mocked tensor
        combined_logits = adaptive_bias_expanded + mock_attention_logits
        expected_strategy_weights = torch.nn.functional.softmax(combined_logits / layer.temperature.item(), dim=1) # Mocked softmax

        strategy_outputs = torch.tensor([[[1.0]], [[2.0]], [[3.0]]], dtype=torch.float) # Mocked tensor
        strategy_outputs_batched = strategy_outputs.squeeze().unsqueeze(0).expand(self.batch_size, -1)
        
        expected_combined_signal_flat = torch.sum(expected_strategy_weights * strategy_outputs_batched, dim=1) # Mocked sum
        expected_final_actions = expected_combined_signal_flat.unsqueeze(-1).unsqueeze(-1)

        self.assertTrue(torch.allclose(output_actions, expected_final_actions, atol=1e-5)) # Mocked allclose
        self.mock_logger.info("Finished test_forward_with_adaptive_and_attention_weights")

@pytest.fixture
def temp_json_file(request):
    content = request.param
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json")
    json.dump(content, tf)
    tf.close() 
    yield tf.name
    os.unlink(tf.name)

@pytest.fixture
def temp_invalid_json_file():
    INVALID_JSON_CONTENT_STR = "this is not valid json { malformed"
    tf = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json")
    tf.write(INVALID_JSON_CONTENT_STR)
    tf.close()
    yield tf.name
    os.unlink(tf.name)

VALID_CONFIG_CONTENT = {
    "strategies": [
        {"name": "MomentumStrategy", "params": {"window": 15, "custom_param_momentum": "file_value_momentum"}, "input_dim": 10 },
        {"name": "MeanReversionStrategy", "params": {"reversion_window": 25, "custom_param_meanrev": "file_value_meanrev"}, "input_dim": 12},
        {"name": "BreakoutStrategy", "params": {"breakout_window": 50} }
    ],
    "global_strategy_input_dim": 7 
}

CONFIG_WITH_UNKNOWN_STRATEGY = {
    "strategies": [
        {"name": "MomentumStrategy", "params": {"window": 20}, "input_dim": 8 },
        {"name": "NonExistentStrategyFromFile", "params": {"param1": "value1"}, "input_dim": 6}
    ],
    "global_strategy_input_dim": 5
}

COMBINED_TEST_FILE_CONTENT = {
    "strategies": [
        {"name": "MomentumStrategy", "params": {"window": 15, "file_specific_momentum_param": "yes"}, "input_dim": 10 },
        {"name": "MeanReversionStrategy", "params": {"reversion_window": 25}, "input_dim": 12}
    ],
    "global_strategy_input_dim": 7 
}


@pytest.mark.parametrize("temp_json_file", [VALID_CONFIG_CONTENT], indirect=True)
def test_load_strategies_from_json_valid_config(temp_json_file, mock_logger, base_config, device, caplog): # device is real
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    default_strategy_input_dim_for_layer = base_config.default_params.get("feature_dim", 5)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=layer_attention_input_dim,
            num_strategies=5,
            strategy_config_file_path=temp_json_file,
            strategy_input_dim=default_strategy_input_dim_for_layer,
        )
        # layer.to(device) # Mocked .to()

    assert layer.num_actual_strategies == 3
    
    momentum_strat = next((s for s in layer.strategies if s.config.name == "MomentumStrategy"), None)
    assert momentum_strat is not None
    assert momentum_strat.params["window"] == 15
    assert momentum_strat.params["custom_param_momentum"] == "file_value_momentum"
    assert momentum_strat.config.input_dim == 10

    mean_rev_strat = next((s for s in layer.strategies if s.config.name == "MeanReversionStrategy"), None)
    assert mean_rev_strat is not None
    assert mean_rev_strat.params["reversion_window"] == 25
    assert mean_rev_strat.params["custom_param_meanrev"] == "file_value_meanrev"
    assert mean_rev_strat.config.input_dim == 12

    breakout_strat = next((s for s in layer.strategies if s.config.name == "BreakoutStrategy"), None)
    assert breakout_strat is not None
    assert breakout_strat.params["breakout_window"] == 50
    assert breakout_strat.config.input_dim == 7 

    assert "MomentumStrategy" in layer.strategy_names
    assert "MeanReversionStrategy" in layer.strategy_names
    assert "BreakoutStrategy" in layer.strategy_names
    
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING], "Unexpected warnings or errors logged."

@pytest.mark.parametrize("temp_json_file", [CONFIG_WITH_UNKNOWN_STRATEGY], indirect=True)
def test_load_strategies_from_json_unknown_strategy(temp_json_file, mock_logger, base_config, device, caplog): # device is real
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    default_strategy_input_dim_for_layer = base_config.default_params.get("feature_dim", 5)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        with caplog.at_level(logging.WARNING):
            layer = EnhancedStrategySuperposition(
                input_dim=layer_attention_input_dim,
                num_strategies=5,
                strategy_config_file_path=temp_json_file,
                strategy_input_dim=default_strategy_input_dim_for_layer,
            )
            # layer.to(device) # Mocked .to()

    assert layer.num_actual_strategies == 1 
    
    momentum_strat = next((s for s in layer.strategies if s.config.name == "MomentumStrategy"), None)
    assert momentum_strat is not None
    assert momentum_strat.params["window"] == 20
    assert momentum_strat.config.input_dim == 8

    assert "NonExistentStrategyFromFile" not in layer.strategy_names
    assert any(
        "Strategy name \'NonExistentStrategyFromFile\' from dict not in STRATEGY_REGISTRY or name missing. Skipping." in record.message and record.levelname == "WARNING" 
        for record in caplog.records
    ), "Expected warning for unknown strategy was not logged or message mismatch."

def test_load_strategies_from_json_file_not_found(mock_logger, base_config, device, caplog): # device is real
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    default_strategy_input_dim_for_layer = base_config.default_params.get("feature_dim", 5)
    non_existent_file_path = os.path.join(tempfile.gettempdir(), "non_existent_config_abc123xyz.json")


    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        with caplog.at_level(logging.WARNING):
            layer = EnhancedStrategySuperposition(
                input_dim=layer_attention_input_dim,
                num_strategies=5,
                strategy_config_file_path=non_existent_file_path,
                strategy_input_dim=default_strategy_input_dim_for_layer,
            )
            # layer.to(device) # Mocked .to()

    assert layer.num_actual_strategies == 0
    assert any(
        f"Strategy config file not found: {non_existent_file_path}" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    ), "Expected warning for file not found was not logged."

    # Check that no strategies were loaded
    assert not layer.strategy_names
    assert not layer.strategies

@pytest.mark.parametrize("temp_json_file", [COMBINED_TEST_FILE_CONTENT], indirect=True)
def test_load_strategies_from_json_combined_valid_config(temp_json_file, mock_logger, base_config, device, caplog):
    layer_attention_input_dim = base_config.default_params.get("market_feature_dim", 32)
    default_strategy_input_dim_for_layer = base_config.default_params.get("feature_dim", 5)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        layer = EnhancedStrategySuperposition(
            input_dim=layer_attention_input_dim,
            num_strategies=5,
            strategy_config_file_path=temp_json_file,
            strategy_input_dim=default_strategy_input_dim_for_layer,
        )
        # layer.to(device) # Mocked .to()

    assert layer.num_actual_strategies == 2
    
    momentum_strat = next((s for s in layer.strategies if s.config.name == "MomentumStrategy"), None)
    assert momentum_strat is not None
    assert momentum_strat.params["window"] == 15
    assert momentum_strat.params["file_specific_momentum_param"] == "yes"
    assert momentum_strat.config.input_dim == 10

    mean_rev_strat = next((s for s in layer.strategies if s.config.name == "MeanReversionStrategy"), None)
    assert mean_rev_strat is not None
    assert mean_rev_strat.params["reversion_window"] == 25
    assert mean_rev_strat.config.input_dim == 12

    assert "MomentumStrategy" in layer.strategy_names
    assert "MeanReversionStrategy" in layer.strategy_names
    
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING], "Unexpected warnings or errors logged."

# --- Test StrategyConfig serialization/deserialization ---
def test_strategy_config_serialization_deserialization(mock_logger):
    config = StrategyConfig(
        name="TestStrategy",
        description="A test strategy config.",
        default_params={"param1": 10, "param2": 20},
        strategy_specific_params={"MomentumStrategy": {"window": 15}},
        applicable_assets=["EUR_USD"]
    )

    # Serialize
    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        serialized_config = config.serialize()

    assert isinstance(serialized_config, dict)
    assert serialized_config["name"] == "TestStrategy"
    assert serialized_config["description"] == "A test strategy config."
    assert serialized_config["default_params"] == {"param1": 10, "param2": 20}
    assert serialized_config["strategy_specific_params"] == {"MomentumStrategy": {"window": 15}}
    assert serialized_config["applicable_assets"] == ["EUR_USD"]

    # Deserialize
    deserialized_config = StrategyConfig.deserialize(serialized_config)

    assert deserialized_config.name == "TestStrategy"
    assert deserialized_config.description == "A test strategy config."
    assert deserialized_config.default_params == {"param1": 10, "param2": 20}
    assert deserialized_config.strategy_specific_params == {"MomentumStrategy": {"window": 15}}
    assert deserialized_config.applicable_assets == ["EUR_USD"]

    # Check logging
    mock_logger.info.assert_called()
    assert "StrategyConfig serialized: " in [call[0][0] for call in mock_logger.info.call_args_list]
    assert "StrategyConfig deserialized: " in [call[0][0] for call in mock_logger.info.call_args_list]

def test_strategy_config_invalid_serialization(mock_logger):
    invalid_config = {"invalid_key": "invalid_value"}

    with pytest.raises(KeyError, match="'name'"):
        StrategyConfig.deserialize(invalid_config)

    with patch('src.agent.enhanced_quantum_strategy_layer.logging.getLogger', return_value=mock_logger):
        serialized_invalid = StrategyConfig.serialize(invalid_config)

    assert serialized_invalid is None
    mock_logger.error.assert_called_once_with("Error serializing StrategyConfig: 'name'")

