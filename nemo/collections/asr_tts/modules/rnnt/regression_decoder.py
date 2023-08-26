from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from nemo.collections.asr.modules import RNNTDecoder
from nemo.core.classes import typecheck
from nemo.core.neural_types import LengthsType, LossType, NeuralType, VoidType
from nemo.utils import logging


class TransducerRegressionDecoder(RNNTDecoder):
    def __init__(self, prednet: Dict[str, Any]):
        super().__init__(prednet=prednet, vocab_size=2)
        self.proj = nn.Linear(prednet["in_features"], self.pred_hidden, bias=False)

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('B', 'T', 'C'), VoidType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('D', 'B', 'D'), VoidType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType((('D', 'B', 'D')), VoidType(), optional=True)],  # must always be last
        }

    @typecheck()
    def forward(self, targets, target_length, states=None):
        decoded, states = self.predict(targets, states=states)  # (B, U, D)
        decoded = decoded.transpose(1, 2)  # (B, D, U)
        return decoded, target_length, states

    def predict(
        self, targets: torch.Tensor, states: Optional[torch.Tensor] = None, add_sos=True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # logging.warning(f"{targets.shape}")
        device = targets.device
        dtype = targets.dtype
        batch_size = targets.shape[0]
        num_features = targets.shape[-1]
        if add_sos:
            targets = torch.cat(
                (torch.zeros((batch_size, 1, num_features), dtype=dtype, device=device), targets), dim=1
            )
        # Forward step through RNN
        targets = targets.transpose(0, 1)  # (U + 1, B, H)
        # logging.warning(f"{targets.shape}")
        g, states = self.prediction["dec_rnn"](self.proj(targets), states)
        g = g.transpose(0, 1)  # (B, U + 1, H)
        return g, states
