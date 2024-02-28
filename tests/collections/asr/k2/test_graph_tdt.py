import random

import numpy as np
import pytest
import torch

from nemo.collections.asr.losses.rnnt import TDTLossPytorch
from nemo.collections.asr.parts.k2.tdt_transducer import GraphTDTTransducerLoss
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba


class TestGraphTDT:
    @pytest.mark.unit
    def test_loss_random(self):
        # numba_utils.skip_numba_cuda_test_if_unsupported(__NUMBA_MINIMUM_VERSION__)
        device = torch.device("cuda:0")
        batch_size, max_time, U, V = 4, 8, 4, 8  # here V is number of non blank labels
        durations = [0, 1, 2, 4]
        sigma = 0.00

        encoder_lengths = torch.full([batch_size], fill_value=max_time, device=device, dtype=torch.long)
        label_lengths = torch.full([batch_size], fill_value=U - 1, device=device, dtype=torch.long)

        logits = torch.rand([batch_size, max_time, U, V + 1 + len(durations)], device=device, requires_grad=True)
        logits2 = logits.clone().detach()
        logits2.requires_grad_(True)

        labels = torch.tensor(
            [[random.randrange(0, V) for i in range(U - 1)] for j in range(batch_size)],
            device=device,
            dtype=torch.long,
        )

        # tdt_numba = TDTLossNumba(blank=V, reduction='sum', durations=durations, sigma=sigma)
        # pt_cost, pt_grads = wrap_and_call(fn_pt, acts, labels, device)

        tdt_pytorch = TDTLossPytorch(
            blank=V, reduction='none', durations=durations, sigma=sigma
        )  # ag for automatic gradient computation
        tdt_graph = GraphTDTTransducerLoss(blank=V, durations=durations, sigma=sigma)
        # ag_cost, ag_grads = wrap_and_call(fn_ag, acts, labels, device)

        etalon_loss_value = tdt_pytorch(acts=logits, act_lens=encoder_lengths, labels=labels, label_lens=label_lengths)
        graph_loss_value = tdt_graph(acts=logits2, act_lens=encoder_lengths, labels=labels, label_lens=label_lengths)
        assert torch.allclose(etalon_loss_value, graph_loss_value, rtol=1e-4, atol=1e-4)

        etalon_loss_value.sum().backward()
        graph_loss_value.sum().backward()
        assert torch.allclose(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)
