from typing import Optional

import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.core.classes import Typing


class GreedyBatchedFactorizedTransducerRegressionInfer(Typing):
    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        if self.preserve_alignments:
            raise NotImplementedError

    # @typecheck()
    def forward(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            ???.
        """
        with self.decoder.as_frozen(), self.joint.as_frozen(), torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # [B, D, T] -> [B, T, D]
            device = encoder_output.device
            # Initialize list of Hypothesis
            batch_size, time, emb_size = encoder_output.shape
            hypotheses = [[] for _ in range(batch_size)]

            num_features = self.decoder.num_features  # todo: impl

            hidden = None
            is_start = True
            time_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            last_predictions = torch.full(
                [batch_size, num_features], fill_value=self._blank_index, dtype=torch.long, device=device
            )
            is_active = torch.full([batch_size], fill_value=True, dtype=torch.bool, device=device)
            indices = torch.arange(batch_size, dtype=torch.long, device=device)
            active_indices = indices
            while active_indices.shape[0] > 0:
                embeddings_selected = encoder_output[active_indices, time_indices[is_active]].unsqueeze(1)
                current_batch_size = active_indices.shape[0]
                if is_start:
                    decoder_output, current_hidden = self.decoder.predict_first(
                        hidden, batch_size=current_batch_size
                    )  # todo: impl
                    is_start = False
                else:
                    decoder_output, current_hidden = self.decoder.predict(
                        last_predictions, hidden, batch_size=current_batch_size
                    )  # todo: impl
                blank_logits, predictions = self.joint.joint(embeddings_selected, decoder_output)
                predictions = predictions.squeeze(1).squeeze(1)

                for current_batch_i, source_batch_i in enumerate(active_indices):
                    # label = labels[current_batch_i].item()
                    is_blank = blank_logits[current_batch_i].item() >= 0  # sigmoid acivation => 0 -> 0.5
                    if not is_blank:
                        hypotheses[source_batch_i].append(
                            predictions[current_batch_i].copy()
                        )  # TODO: can be more effective
                        # hypotheses[source_batch_i].timestep.append(time_indices[source_batch_i].item())

                is_active_prev = is_active
                blank_mask = blank_logits >= 0
                time_indices[is_active_prev] += blank_mask
                is_active = time_indices < encoded_lengths
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
                    blank_logits_masked >= 0, last_predictions[local_mask], predictions[local_mask]
                )
                active_indices = indices[is_active]
                if active_indices.shape[0] < batch_size:
                    if isinstance(hidden, tuple):
                        hidden = hidden[0][:, local_mask].contiguous(), hidden[1][:, local_mask].contiguous()
                    else:
                        hidden = hidden[:, local_mask]
        return hypotheses
