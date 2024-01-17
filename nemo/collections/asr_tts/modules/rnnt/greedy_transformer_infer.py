from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInfer
from nemo.collections.asr.parts.utils import rnnt_utils
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
            raise NotImplementedError("`partial_hypotheses` support is not implemented")

        batch_size, max_time, _ = x.shape

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # Initialize empty hypotheses and all necessary tensors
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size, init_length=max_time, device=x.device, float_dtype=x.dtype
        )
        time_indices = torch.zeros([batch_size], dtype=torch.long, device=device)  # always of batch_size
        active_indices = torch.arange(batch_size, dtype=torch.long, device=device)  # initial: all indices
        labels = torch.full([batch_size], fill_value=self._blank_index, dtype=torch.long, device=device)
        state: Optional[TransformerDecoderState] = None

        # init additional structs for hypotheses: last decoder state, alignments, frame_confidence
        # last_decoder_state = [None for _ in range(batch_size)] - not implemented for Transformer

        alignments: Optional[rnnt_utils.BatchedAlignments]
        if self.preserve_alignments or self.preserve_frame_confidence:
            alignments = rnnt_utils.BatchedAlignments(
                batch_size=batch_size,
                logits_dim=self.joint.num_classes_with_blank,
                init_length=max_time * 2,  # blank for each timestep + text tokens
                device=x.device,
                float_dtype=x.dtype,
                store_alignments=self.preserve_alignments,
                store_frame_confidence=self.preserve_frame_confidence,
            )
        else:
            alignments = None

        # loop while there are active indices
        while (current_batch_size := active_indices.shape[0]) > 0:
            # stage 1: get decoder (prediction network) output
            if self.stateful_decoding:
                decoder_output, state, *_ = self.decoder.predict_step(input_ids=labels, state=state)
                decoder_output = decoder_output.transpose(1, 2)
            else:
                # use full batch of full transcripts with corresponding lengths
                decoder_output, *_ = self._pred_step(
                    batched_hyps.transcript[active_indices], batched_hyps.current_lengths[active_indices],
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

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
            if alignments is not None:
                alignments.add_results_(
                    active_indices=active_indices,
                    time_indices=time_indices[active_indices],
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=torch.tensor(self._get_confidence(logits), device=device)
                    if self.preserve_frame_confidence
                    else None,
                )
            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            advance_mask = torch.logical_and(blank_mask, (time_indices[active_indices] + 1 < out_len[active_indices]))
            while advance_mask.any():
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
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                if self.use_sampling:
                    more_labels = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
                    more_scores = logits[torch.arange(0, logits.shape[0], device=device), more_labels]
                else:
                    more_scores, more_labels = logits.max(-1)
                labels[advance_mask] = more_labels
                scores[advance_mask] = more_scores
                if alignments is not None:
                    alignments.add_results_(
                        active_indices=advance_indices,
                        time_indices=time_indices[advance_indices],
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=torch.tensor(self._get_confidence(logits), device=device)
                        if self.preserve_frame_confidence
                        else None,
                    )
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

                # TODO: implement for transformer
                # select states for hyps that became inactive (is it necessary?)
                # this seems to be redundant, but used in the `loop_frames` output
                # inactive_global_indices = active_indices[blank_mask]
                # inactive_inner_indices = torch.arange(current_batch_size, device=device, dtype=torch.long)[blank_mask]
                # for idx, batch_idx in zip(inactive_global_indices.cpu().numpy(), inactive_inner_indices.cpu().numpy()):
                #     last_decoder_state[idx] = self.decoder.batch_select_state(state, batch_idx)

                # update active indices and state
                active_indices = active_indices[non_blank_mask]
                if self.stateful_decoding:
                    state.filter_(active_mask=non_blank_mask)
                # state = self.decoder.mask_select_states(state, non_blank_mask)
            # store hypotheses
            batched_hyps.add_results_(
                active_indices, labels, time_indices[active_indices].clone(), scores,
            )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    torch.logical_and(
                        labels != self._blank_index,
                        batched_hyps.last_timestep_lasts[active_indices] >= self.max_symbols,
                    ),
                    batched_hyps.last_timestep[active_indices] == time_indices[active_indices],
                )
                if force_blank_mask.any():
                    # forced blank is not stored in the alignments following the original implementation
                    time_indices[active_indices[force_blank_mask]] += 1  # emit blank => advance time indices
                    # elements with time indices >= out_len become inactive, remove them from batch
                    still_active_mask = time_indices[active_indices] < out_len[active_indices]
                    if self.stateful_decoding:
                        state.filter_(active_mask=still_active_mask)
                    active_indices = active_indices[still_active_mask]
                    labels = labels[still_active_mask]
                    state = self.decoder.mask_select_states(state, still_active_mask)

        hyps = rnnt_utils.batched_hyps_to_hypotheses(batched_hyps, alignments)
        # TODO: implement for transformer
        # # preserve last decoder state (is it necessary?)
        # for i, last_state in enumerate(last_decoder_state):
        #     # assert last_state is not None
        #     hyps[i].dec_state = last_state
        return hyps

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
