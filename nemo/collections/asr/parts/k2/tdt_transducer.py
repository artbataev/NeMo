# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from typing import Union

import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.k2.graph_transducer import GraphRnntLoss, force_float32_context
from nemo.core.utils.k2_guard import k2


class GraphTDTTransducerLoss(GraphRnntLoss):
    def __init__(
        self,
        blank: int,
        durations: list[int],
        clamp: float = -1,
        sigma: float = 0.0,
        omega: float = 0.0,
        use_grid_implementation=True,
        connect_composed=False,
        double_scores=False,
        cast_to_float32=False,
        return_graph=False,
    ):
        """
        Init method

        Args:
            blank: blank label index
            use_grid_implementation: Whether to use the grid implementation (Grid-Transducer).
            connect_composed: Connect graph after composing unit and temporal schemas
                (only for Compose-Transducer). `connect` operation is slow, it is useful for visualization,
                but not necessary for loss computation.
            double_scores: Use calculation of loss in double precision (float64) in the lattice.
                Does not significantly affect memory usage since the lattice is ~V/2 times smaller than the joint tensor.
            cast_to_float32: Force cast joint tensor to float32 before log-softmax calculation.
            return_graph: Return graph (along with loss) from `forward` function
        """
        super().__init__(
            blank=blank,
            use_grid_implementation=use_grid_implementation,
            connect_composed=connect_composed,
            double_scores=double_scores,
            cast_to_float32=cast_to_float32,
            return_graph=return_graph,
        )
        if clamp > 0.0:
            raise NotImplementedError
        self.sigma = sigma
        self.omega = omega
        self.durations = durations
        assert durations[0] == 0
        assert durations[1] == 1

    """
    Graph-TDT Transducer loss
    """

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        raise NotImplementedError

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        raise NotImplementedError

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        """
        Construct the RNN-T lattice directly (Grid-Transducer).

        Args:
            units_tensor: 1d tensor with text units
            num_frames: length of the sequence (number of frames)
            vocab_size: number of total labels (vocab size including blank)

        Returns:
            transducer lattice (k2.Fsa).
            Labels: <unit>:<frame_index>:<unit_position>:<unit_durations>
                    (k2.Fsa: labels, aux_labels, unit_positions, unit_durations)
        """
        blank_id = self.blank
        text_length = units_tensor.shape[0]
        device = units_tensor.device
        # all_durations = torch.tensor(list(self.durations), device=device, dtype=torch.long)
        num_grid_states = num_frames * (text_length + 1)
        num_forward_arcs_rnnt = (num_frames - 1) * (text_length + 1)
        num_text_arcs_rnnt = text_length * num_frames
        # num_extra_forward_arcs = ...
        arcs = torch.zeros(
            (num_forward_arcs_rnnt * (len(self.durations) - 1) + num_text_arcs_rnnt * len(self.durations) + 2, 4),
            dtype=torch.int32,
            device=device,
        )
        arc_durations = torch.zeros_like(arcs[:, 0])
        # blank transitions
        # i, i+<text_len + 1>, 0 <blank>, i / <text_len+1>, i % <text_len + 1>
        # from_states = (
        #     torch.arange(num_grid_states, dtype=torch.int32, device=device)
        #     .reshape(num_frames, text_length + 1)[:-1, :]
        #     .flatten()
        # )
        # to_states = from_states + (text_length + 1)
        cur_last = 0
        # arcs[:num_forward_arcs_rnnt, 0] = from_states
        # arcs[:num_forward_arcs_rnnt, 1] = to_states
        # arcs[:num_forward_arcs_rnnt, 2] = blank_id
        # durations[:num_forward_arcs_rnnt] = 1
        # cur_last += num_forward_arcs_rnnt
        for cur_duration in self.durations[1:]:
            if cur_duration > num_frames:
                break
            num_extra = (num_frames - cur_duration) * (text_length + 1)
            from_states = (
                torch.arange(num_grid_states, dtype=torch.int32, device=device)
                .reshape(num_frames, text_length + 1)[:-cur_duration, :]
                .flatten()
            )
            to_states = from_states + (text_length + 1) * cur_duration
            arcs[cur_last : cur_last + num_extra, 0] = from_states
            arcs[cur_last : cur_last + num_extra, 1] = to_states
            arcs[cur_last : cur_last + num_extra, 2] = blank_id
            arc_durations[cur_last : cur_last + num_extra] = cur_duration
            cur_last += num_extra
            if cur_duration > 1:
                arcs[cur_last : cur_last + 1, 0] = num_grid_states - 1 - (cur_duration - 1) * (text_length + 1)
                arcs[cur_last : cur_last + 1, 1] = num_grid_states
                arcs[cur_last : cur_last + 1, 2] = blank_id
                arc_durations[cur_last : cur_last + 1] = cur_duration
                cur_last += 1

        # text arcs
        from_states = (
            torch.arange(num_grid_states, dtype=torch.int32, device=device)
            .reshape(num_frames, text_length + 1)[:, :-1]
            .flatten()
        )
        to_states = from_states + 1
        ilabels = units_tensor.expand(num_frames, -1).flatten()
        arcs[cur_last : cur_last + num_text_arcs_rnnt, 0] = from_states
        arcs[cur_last : cur_last + num_text_arcs_rnnt, 1] = to_states
        arcs[cur_last : cur_last + num_text_arcs_rnnt, 2] = ilabels
        arc_durations[cur_last : cur_last + num_text_arcs_rnnt] = 0
        cur_last += num_text_arcs_rnnt

        for cur_duration in self.durations[1:]:
            if cur_duration > num_frames:
                break
            num_extra = text_length * (num_frames - cur_duration)
            from_states = (
                torch.arange(num_grid_states, dtype=torch.int32, device=device)
                .reshape(num_frames, text_length + 1)[:-cur_duration, :-1]
                .flatten()
            )
            # assert from_states.shape[0] == num_extra
            to_states = from_states + 1 + (text_length + 1) * cur_duration
            ilabels = units_tensor.expand(num_frames - cur_duration, -1).flatten()
            arcs[cur_last : cur_last + num_extra, 0] = from_states
            arcs[cur_last : cur_last + num_extra, 1] = to_states
            arcs[cur_last : cur_last + num_extra, 2] = ilabels
            arc_durations[cur_last : cur_last + num_extra] = cur_duration
            cur_last += num_extra
            # if cur_duration >= 1:
            arcs[cur_last : cur_last + 1, 0] = num_grid_states - 2 - (cur_duration - 1) * (text_length + 1)
            arcs[cur_last : cur_last + 1, 1] = num_grid_states
            arcs[cur_last : cur_last + 1, 2] = units_tensor[-1]
            arc_durations[cur_last : cur_last + 1] = cur_duration
            cur_last += 1

        arcs = arcs[: cur_last + 2]
        arc_durations = arc_durations[: cur_last + 2]
        # last 2 states
        # last blank
        arcs[-2, :3] = torch.tensor((num_grid_states - 1, num_grid_states, blank_id), dtype=torch.int32, device=device)
        arc_durations[-2] = 1
        # transition to final state - special for k2
        arcs[-1, :3] = torch.tensor((num_grid_states, num_grid_states + 1, -1), dtype=torch.int32, device=device)

        # sequence indices, time indices
        frame_indices = torch.div(
            arcs[:, 0], (text_length + 1), rounding_mode="floor"
        )  # arcs[:, 0] // (text_length + 1)
        unit_positions = arcs[:, 0] % (text_length + 1)
        # last state: final
        frame_indices[-1] = -1
        unit_positions[-1] = -1

        # relabel
        # instead of using top sort (extremely expensive) k2.top_sort(rnnt_graph)
        arcs[:-2, 0] = self.relabel_states(arcs[:-2, 0], text_length + 1, num_frames)
        arcs[:-3, 1] = self.relabel_states(arcs[:-3, 1], text_length + 1, num_frames)

        # sort by start state - required in k2
        indices = torch.argsort(arcs[:, 0], dim=0)
        rnnt_graph = k2.Fsa(arcs[indices], aux_labels=frame_indices[indices])
        rnnt_graph.unit_positions = unit_positions[indices]
        rnnt_graph.durations = arc_durations[indices]
        return rnnt_graph

    def forward(
        self, acts: torch.Tensor, labels: torch.Tensor, act_lens: torch.Tensor, label_lens: torch.Tensor,
    ):
        raise NotImplementedError
