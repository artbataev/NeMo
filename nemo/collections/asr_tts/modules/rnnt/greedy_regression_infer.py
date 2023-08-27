from typing import Dict, Optional

import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr_tts.modules.rnnt.joint_regression import FactorizedRegressionJoint
from nemo.collections.asr_tts.modules.rnnt.regression_decoder import TransducerRegressionDecoder
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import ElementType, NeuralType, VoidType
from nemo.utils import logging


class BatchedHyps:
    def __init__(self, batch_size: int, init_length: int, num_features: int, device=None, float_dtype=None):
        self.y_sequence = torch.zeros((batch_size, init_length, num_features), device=device, dtype=float_dtype)
        self.timestep = torch.zeros((batch_size, init_length), device=device, dtype=torch.long)
        # self.scores = torch.zeros(batch_size, device=device, dtype=float_dtype)
        self.max_length = init_length
        self.num_features = num_features
        self.indices = torch.zeros(batch_size, device=device, dtype=torch.long)

    def _allocate_more(self):
        """Allocate twice"""
        self.y_sequence = torch.cat((self.y_sequence, torch.zeros_like(self.y_sequence)), dim=1)
        self.timestep = torch.cat((self.timestep, torch.zeros_like(self.timestep)), dim=1)
        self.max_length *= 2

    def add_results(self, active_indices, features, time_indices):
        # we assume that all tensors have the same first dimension, and labels are non-blanks
        if active_indices.shape[0] == 0:
            return  # nothing to add
        if self.indices.max().item() >= self.max_length:
            self._allocate_more()
        # self.scores[active_indices] += scores
        self.y_sequence.view(-1, self.num_features)[
            active_indices * self.max_length + self.indices[active_indices]
        ] = features.clone().detach()
        self.timestep.view(-1)[
            active_indices * self.max_length + self.indices[active_indices]
        ] = time_indices.clone().detach()
        self.indices[active_indices] += 1


class GreedyBatchedFactorizedTransducerRegressionInfer(Typing):
    def __init__(
        self, decoder_model: TransducerRegressionDecoder, joint_model: FactorizedRegressionJoint, num_features: int
    ):
        self.decoder = decoder_model
        self.joint = joint_model
        self.num_features = num_features

    @property
    def input_types(self) -> Dict[str, NeuralType]:
        return {
            "encoder_output": NeuralType(('B', 'T', 'D'), VoidType()),
            "encoder_lengths": NeuralType(tuple('B'), VoidType()),
            # "max_lengths": NeuralType((), VoidType()),
        }

    # @typecheck()
    def greedy_decode(
        self, encoder_output: torch.Tensor, encoder_lengths: torch.Tensor, max_decode_length: int,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            ???.
        """
        with self.decoder.as_frozen(), self.joint.as_frozen(), torch.inference_mode():
            # Apply optional preprocessing
            device = encoder_output.device
            # Initialize list of Hypothesis
            batch_size, time, emb_size = encoder_output.shape
            # hypotheses = [[] for _ in range(batch_size)]
            hypotheses = BatchedHyps(
                batch_size, time, self.num_features, device=device, float_dtype=encoder_output.dtype
            )

            hidden = None
            # is_start = True
            time_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            # start with zeros
            last_predictions = torch.zeros([batch_size, self.num_features], device=device)
            # TODO: alternatives for zeros for first prediction
            is_active = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)
            indices = torch.arange(batch_size, dtype=torch.long, device=device)
            active_indices = indices

            while active_indices.shape[0] > 0 and hypotheses.indices.max() < max_decode_length:
                embeddings_selected = encoder_output[active_indices, time_indices[is_active]].unsqueeze(1)
                decoder_output, current_hidden = self.decoder.predict(
                    last_predictions.unsqueeze(1), hidden, add_sos=False
                )
                logits = self.joint.joint(embeddings_selected, decoder_output)
                blank_logits = logits[..., 0].squeeze(1).squeeze(1)
                predictions = logits[..., 1:].squeeze(1).squeeze(1)
                del logits

                is_active_prev = is_active
                blank_mask = blank_logits >= 0
                not_blank_mask = ~blank_mask
                hypotheses.add_results(
                    active_indices[not_blank_mask],
                    predictions[not_blank_mask],
                    time_indices[active_indices[not_blank_mask]],
                    # scores[not_blank_mask],
                )
                time_indices[is_active_prev] += blank_mask
                is_active = time_indices < encoder_lengths
                local_mask = is_active[is_active_prev]
                if hidden is None:
                    if isinstance(current_hidden, tuple):
                        hidden = (torch.zeros_like(current_hidden[0]), torch.zeros_like(current_hidden[1]))
                    else:
                        hidden = torch.zeros_like(current_hidden)
                if isinstance(hidden, tuple):
                    hidden0 = torch.where(blank_mask[None, :, None], hidden[0], current_hidden[0])
                    hidden1 = torch.where(blank_mask[None, :, None], hidden[1], current_hidden[1])
                    hidden = (hidden0.contiguous(), hidden1.contiguous())
                    # hidden = (hidden0, hidden1)
                else:
                    hidden = torch.where(blank_mask[None, :, None], hidden, current_hidden)
                # labels = labels.unsqueeze(-1)
                blank_logits_masked = blank_logits[local_mask]
                last_predictions = torch.where(
                    (blank_logits_masked >= 0).unsqueeze(1), last_predictions[local_mask], predictions[local_mask],
                )
                active_indices = indices[is_active]
                if active_indices.shape[0] < batch_size:
                    if isinstance(hidden, tuple):
                        hidden = hidden[0][:, local_mask].contiguous(), hidden[1][:, local_mask].contiguous()
                    else:
                        hidden = hidden[:, local_mask]
        return hypotheses
