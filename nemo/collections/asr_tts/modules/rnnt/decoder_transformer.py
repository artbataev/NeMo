import copy
from typing import Any, Dict, List, Optional

import torch

from nemo.collections.asr.modules import SpectrogramAugmentation
from nemo.collections.nlp.modules.common.transformer import TransformerEmbedding, TransformerEncoder
from nemo.core import NeuralModule
from nemo.core.classes import typecheck
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType
from nemo.utils import logging


class TransformerDecoderState:
    transformer_state: torch.Tensor  # List[torch.Tensor]
    lengths: torch.Tensor

    def __init__(self, transformer_state: List[torch.Tensor], prev_state: Optional["TransformerDecoderState"] = None):
        self.transformer_state = torch.stack(transformer_state)
        batch_size = transformer_state[0].shape[0]
        device = transformer_state[0].device
        if prev_state is None:
            self.lengths = torch.ones([batch_size], device=device, dtype=torch.long)
        else:
            self.lengths = prev_state.lengths + 1
            # fix added state at the last index
            self.transformer_state[
                :, torch.arange(batch_size, device=device), self.lengths - 1
            ] = self.transformer_state[:, :, -1].clone()
            # for state in self.transformer_state:
            #     #  clone is necessary here to avoid single-element tensor problems
            #     state[torch.arange(batch_size), self.lengths - 1] = state[:, -1].clone()

    def filter_(self, active_mask: torch.Tensor):
        if active_mask.sum() == active_mask.shape[0]:
            return  # nothing to filter
        assert active_mask.shape[0] == self.lengths.shape[0]
        self.transformer_state = self.transformer_state[:, active_mask]
        # for i, state in enumerate(self.transformer_state):
        #     self.transformer_state[i] = state[active_mask]
        self.lengths = self.lengths[active_mask]
        self._fix_shape()

    def _fix_shape(self):
        # empty state
        if self.lengths.shape[0] == 0:
            return
        max_length = self.lengths.max()
        if max_length >= self.transformer_state[0].shape[1]:
            return  # nothing to fix
        self.transformer_state = torch.narrow(self.transformer_state, dim=2, start=0, length=max_length)
        # for i, state in enumerate(self.transformer_state):
        #     self.transformer_state[i] = torch.narrow(state, dim=1, start=0, length=max_length)

    def reduce_length_(self, blank_mask: torch.Tensor):
        self.lengths -= blank_mask.to(torch.long)
        self._fix_shape()

    def get_mask(self):
        mask = (
            torch.arange(self.transformer_state[0].shape[1], device=self.lengths.device)[None, :]
            < self.lengths[:, None]
        )
        return mask


class RNNTTransformerDecoder(NeuralModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_layer: Dict[str, Any],
        prediction_network: Dict[str, Any],
        spec_augment: Optional[Dict[str, Any]] = None,
        blank_as_pad=True,
    ):
        super().__init__()
        self.blank_as_pad = blank_as_pad
        if not self.blank_as_pad:
            raise NotImplementedError("blank_as_pad=False not implemented")
        embedding_layer = copy.deepcopy(embedding_layer)  # make local copy
        embedding_layer["vocab_size"] = vocab_size + 1
        self.embedding = TransformerEmbedding(**embedding_layer)
        self.spec_augment = SpectrogramAugmentation(**spec_augment) if spec_augment is not None else None
        self.prediction_network = TransformerEncoder(**prediction_network)
        self.blank_idx = vocab_size  # last symbol - as in other RNN-T models

    @property
    def input_types(self):
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    def forward(self, targets: torch.Tensor, target_length: torch.Tensor):
        input_ids = targets
        input_lengths = target_length
        batch_size = input_ids.shape[0]
        blank_prefix = torch.full(
            (batch_size, 1), fill_value=self.blank_idx, device=input_ids.device, dtype=input_ids.dtype
        )
        input_ids = torch.cat((blank_prefix, input_ids), dim=1)
        # `<=` => prefix
        input_mask = torch.arange(input_lengths.max() + 1, device=input_ids.device)[None, :] <= input_lengths[:, None]
        input_embed = self.embedding(input_ids)
        if self.spec_augment is not None and self.training:
            with typecheck.disable_checks():
                input_embed_spaug = self.spec_augment(
                    input_spec=input_embed.transpose(1, 2).detach(), length=input_lengths
                ).transpose(1, 2)
                input_embed[input_embed_spaug == 0.0] = 0.0
        decoder_output = self.prediction_network(
            encoder_states=input_embed, encoder_mask=input_mask, encoder_mems_list=None, return_mems=False,
        )
        return decoder_output.transpose(1, 2), None, input_lengths

    def predict(self, input_ids: torch.Tensor, input_lengths: torch.Tensor):
        # TODO: implement memory
        return self.forward(targets=input_ids, target_length=input_lengths,)

    def predict_step(self, input_ids: torch.Tensor, state: Optional[TransformerDecoderState] = None):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        input_mask = torch.full((batch_size, 1), fill_value=True, device=device, dtype=torch.bool)
        # input_mask = torch.arange(1, device=device)[None, :] <= input_lengths[:, None]
        if state is None:
            input_embed = self.embedding(input_ids.unsqueeze(1), start_pos=0)
            *transformer_state, decoder_output = self.prediction_network(
                encoder_states=input_embed,
                encoder_mask=input_mask,
                encoder_mems_list=None,
                return_mems=True,
                memory_mask=None,
            )
            new_state = TransformerDecoderState(transformer_state=transformer_state, prev_state=None)
            return decoder_output.transpose(1, 2), new_state

        # not first step, state is not None
        input_embed = self.embedding(input_ids.unsqueeze(1), start_pos=state.lengths)
        *transformer_state, decoder_output = self.prediction_network(
            encoder_states=input_embed,
            encoder_mask=input_mask,
            encoder_mems_list=state.transformer_state,
            return_mems=True,
            memory_mask=state.get_mask(),
        )
        next_state = TransformerDecoderState(transformer_state=transformer_state, prev_state=state)
        return decoder_output[:, -1].unsqueeze(-1), next_state
