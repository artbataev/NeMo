from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from nemo.core.classes import Exportable, NeuralModule, typecheck
from nemo.core.neural_types import NeuralType, VoidType


class SimpleLinear(NeuralModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearExportable(NeuralModule, Exportable):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    def forward(self, x):
        return self.linear(x)


class SimpleLinearWithTypes(NeuralModule):
    @property
    def input_types(self) -> Dict[str, Any]:
        return {
            "x": NeuralType(None, VoidType(), optional=True),
        }

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20, bias=False)

    @typecheck()
    def forward(self, x):
        return self.linear(x)


class DummyModuleWithIOTypes(NeuralModule):
    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "x": NeuralType(None, VoidType()),
        }

    @property
    def output_types(self) -> Dict[str, NeuralType]:
        return {
            "x": NeuralType(None, VoidType()),
        }

    def forward(self, x):
        return x


class TestTorchJitCompatibility:
    def test_simple_linear(self):
        module = torch.jit.script(SimpleLinear())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_simple_linear_exportable(self):
        module = torch.jit.script(SimpleLinearExportable())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_simple_linear_with_types(self):
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(x)
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_dummy_module_with_io_types(self):
        module = torch.jit.script(DummyModuleWithIOTypes())
        x = torch.rand(2, 10)
        result = module(x)
        assert result.shape == x.shape
        assert torch.allclose(result, x)

    def test_chain_with_types(self):
        dummy_module = torch.jit.script(DummyModuleWithIOTypes())
        module = torch.jit.script(SimpleLinearWithTypes())
        x = torch.zeros(2, 10)
        result = module(dummy_module(x))
        assert result.shape == (2, 20)
        assert torch.allclose(result, torch.zeros_like(result))
