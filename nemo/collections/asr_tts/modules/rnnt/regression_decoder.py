from typing import List, Optional, Tuple

import torch

from nemo.collections.asr.modules import RNNTDecoder
from nemo.core.classes import typecheck
from nemo.core.neural_types import LengthsType, LossType, NeuralType, VoidType


class TransducerRegressionDecoder(RNNTDecoder):
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
        decoded, states = self.predict(targets, state=states)  # (B, U, D)
        decoded = decoded.transpose(1, 2)  # (B, D, U)
        return decoded, target_length, states

    def predict(
        self,
        targets: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Get device and dtype of current module
        _p = next(self.parameters())
        device = _p.device
        dtype = _p.dtype

        raise NotImplementedError
        # If y is not None, it is of shape [B, U] with dtype long.
        if y is not None:
            if y.device != device:
                y = y.to(device)

            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            # Y is not provided, assume zero tensor with shape [B, 1, H] is required
            # Emulates output of embedding of pad token.
            if batch_size is None:
                B = 1 if state is None else state[0].size(1)
            else:
                B = batch_size

            y = torch.zeros((B, 1, self.pred_hidden), device=device, dtype=dtype)

        # Prepend blank "start of sequence" symbol (zero tensor)
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H), device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()  # (B, U + 1, H)
        else:
            start = None  # makes del call later easier

        # If in training mode, and random_state_sampling is set,
        # initialize state to random normal distribution tensor.
        if state is None:
            if self.random_state_sampling and self.training:
                state = self.initialize_state(y)

        # Forward step through RNN
        y = y.transpose(0, 1)  # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)  # (B, U + 1, H)

        del y, start, state

        # Adapter module forward step
        if self.is_adapter_available():
            g = self.forward_enabled_adapters(g)

        return g, hid
