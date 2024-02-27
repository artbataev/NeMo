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
        durations=None,
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
        if durations is None:
            raise NotImplementedError
        self.durations = durations

    """
    Graph-TDT Transducer loss
    """

    def get_unit_schema(self, units_tensor: torch.Tensor, vocab_size: int) -> "k2.Fsa":
        raise NotImplementedError

    def get_temporal_schema(self, num_frames: int, vocab_size: int, device: torch.device) -> "k2.Fsa":
        raise NotImplementedError

    def get_grid(self, units_tensor: torch.Tensor, num_frames: int, vocab_size: int) -> "k2.Fsa":
        raise NotImplementedError

    def forward(
        self, acts: torch.Tensor, labels: torch.Tensor, act_lens: torch.Tensor, label_lens: torch.Tensor,
    ):
        raise NotImplementedError
