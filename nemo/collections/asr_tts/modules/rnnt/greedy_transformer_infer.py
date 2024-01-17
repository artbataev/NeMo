from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps
from nemo.collections.asr_tts.modules.rnnt.decoder_transformer import RNNTTransformerDecoder, TransformerDecoderState
from nemo.utils import logging


class GreedyBatchedTransformerRNNTInfer(GreedyBatchedRNNTInfer):
    decoder: RNNTTransformerDecoder
    joint: AbstractRNNTJoint

    def __init__(
        self,
        decoder_model: RNNTTransformerDecoder,
        joint_model: AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        stateful_decoding: bool = True,
        use_sampling: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
        )
        self.stateful_decoding = stateful_decoding
        self.use_sampling = use_sampling
        if self.max_symbols is not None and self.max_symbols < 0:
            logging.warning(f"max_symbols={self.max_symbols} < 0 for decoding, setting to None")
            self.max_symbols = None
        if self.max_symbols is None or self.max_symbols > 100:
            logging.warning(
                "Decoding with large or None max_symbols is not recommended, "
                f"currently max_symbols={self.max_symbols}"
            )
        assert isinstance(self.decoder, RNNTTransformerDecoder)
        assert isinstance(self.joint, AbstractRNNTJoint)
        self._greedy_decode = self._greedy_decode_blank_as_pad

    @torch.inference_mode()
    def _greedy_decode_blank_as_pad(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        if self.preserve_frame_confidence:
            raise NotImplementedError("`preserve_frame_confidence` support is not supported")
        if self.preserve_alignments:
            raise NotImplementedError("preserve_alignments` support is not supported")

        # Initialize list of Hypothesis
        batch_size, time, emb_size = x.shape
        x = self.joint.project_encoder(x)
        batched_hyps = BatchedHyps(batch_size=batch_size, init_length=time, device=x.device, float_dtype=x.dtype)
        time_indices = torch.zeros([batch_size], dtype=torch.long, device=device)  # always of batch_size
        active_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # initial: all indices
        state: Optional[TransformerDecoderState] = None
        labels = torch.full([batch_size], fill_value=self._blank_index, dtype=torch.long, device=device)
        while (current_batch_size := active_indices.shape[0]) > 0:
            if self.stateful_decoding:
                decoder_output, state, *_ = self.decoder.predict_step(input_ids=labels, state=state)
                decoder_output = decoder_output.transpose(1, 2)
            else:
                decoder_output, *_ = self._pred_step(
                    batched_hyps.transcript[active_indices], batched_hyps.lengths[active_indices],
                )
            decoder_output = self.joint.project_prednet(decoder_output)

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self._joint_step_after_projection(
                    x[active_indices, time_indices[active_indices]].unsqueeze(1),
                    decoder_output,
                    log_normalize=True if self.preserve_frame_confidence else None,
                )
                .squeeze(1)
                .squeeze(1)
            )
            if self.use_sampling:
                labels = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                scores = logits[torch.arange(0, current_batch_size, device=device), labels]
            else:
                scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            advance_mask = torch.logical_and(blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices]))
            while advance_mask.any():  # .item()?
                advance_indices = active_indices[advance_mask]
                time_indices[advance_indices] += 1
                logits = (
                    self._joint_step_after_projection(
                        x[advance_indices, time_indices[advance_indices]].unsqueeze(1),
                        decoder_output[advance_mask],
                        log_normalize=True if self.preserve_frame_confidence else None,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                if self.use_sampling:
                    more_labels = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                    more_scores = logits[torch.arange(0, logits.shape[0], device=device), more_labels]
                else:
                    more_scores, more_labels = logits.max(-1)
                labels[advance_mask] = more_labels
                scores[advance_mask] = more_scores
                blank_mask = labels == self._blank_index
                advance_mask = torch.logical_and(
                    blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices])
                )

            # stage 3: filter labels and state, store hypotheses
            # the only case, when there are blank labels in predictions - when we found the end for some utterances
            if blank_mask.any():
                non_blank_mask = ~blank_mask
                labels = labels[non_blank_mask]
                scores = scores[non_blank_mask]

                # update active indices and state
                active_indices = active_indices[non_blank_mask]
                if self.stateful_decoding:
                    state.filter_(active_mask=non_blank_mask)

            # store hypotheses
            batched_hyps.add_results_(
                active_indices, labels, time_indices[active_indices].clone(), scores,
            )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                force_blank_mask = torch.logical_and(
                    torch.logical_and(
                        labels != self._blank_index,
                        batched_hyps.last_timestep_lasts[active_indices] >= self.max_symbols,
                    ),
                    batched_hyps.last_timestep[active_indices] == time_indices[active_indices],
                )
                if force_blank_mask.any():
                    time_indices[active_indices[force_blank_mask]] += 1
                    still_active_mask = time_indices[active_indices] < out_len[active_indices]
                    if self.stateful_decoding:
                        state.filter_(active_mask=still_active_mask)
                    active_indices = active_indices[still_active_mask]
                    labels = labels[still_active_mask]
        return rnnt_utils.batched_hyps_to_hypotheses(batched_hyps)

    @torch.no_grad()
    def _pred_step(
        self, labels: torch.Tensor, labels_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        if labels.dtype != torch.long:
            labels = labels.long()
        # TODO: memory
        output, *additional_outputs = self.decoder.predict(
            input_ids=torch.narrow(labels, 1, 0, labels_lengths.max()), input_lengths=labels_lengths,
        )
        return (
            output.transpose(1, 2)[torch.arange(output.shape[0]), labels_lengths].unsqueeze(1),
            *additional_outputs,
        )
