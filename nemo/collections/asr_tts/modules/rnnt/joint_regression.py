from typing import Any, Dict, List, Optional, Union

import torch

from nemo.collections.asr.losses.rnnt import RNNTLossMse
from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LossType, NeuralType, VoidType
from nemo.utils import logging


class FactorizedRegressionJoint(AbstractRNNTJoint):
    loss: RNNTLossMse

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_outputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "decoder_outputs": NeuralType(('B', 'D', 'T'), VoidType()),
            "encoder_lengths": NeuralType(tuple('B'), VoidType()),
            "targets": NeuralType(('B', 'T', 'C'), VoidType()),
            "targets_lengths": NeuralType(tuple('B'), VoidType()),
        }

    def __init__(
        self, jointnet: Dict[str, Any], features_dim: int, fused_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self._loss = None
        self.features_dim = features_dim

        # Required arguments
        self.encoder_hidden = jointnet['encoder_hidden']
        self.pred_hidden = jointnet['pred_hidden']
        self.joint_hidden = jointnet['joint_hidden']
        self.activation = jointnet['activation']

        # Optional arguments
        dropout = jointnet.get('dropout', 0.0)

        self.pred, self.enc, self.joint_net = self._joint_net_modules(
            num_classes=self.features_dim + 1,  # add 1 for blank symbol
            pred_n_hidden=self.pred_hidden,
            enc_n_hidden=self.encoder_hidden,
            joint_n_hidden=self.joint_hidden,
            activation=self.activation,
            dropout=dropout,
        )

        self.fused_batch_size = fused_batch_size

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {
            "loss": NeuralType(elements_type=LossType(), optional=False),
            # "mse": NeuralType(elements_type=LossType(), optional=True),
        }

    def _joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
        """
        Prepare the trainable modules of the Joint Network

        Args:
            num_classes: Number of output classes (vocab size) excluding the RNNT blank token.
            pred_n_hidden: Hidden size of the prediction network.
            enc_n_hidden: Hidden size of the encoder network.
            joint_n_hidden: Hidden size of the joint network.
            activation: Activation of the joint. Can be one of [relu, tanh, sigmoid]
            dropout: Dropout value to apply to joint.
        """
        pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
        enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)

        if activation not in ['relu', 'sigmoid', 'tanh']:
            raise ValueError("Unsupported activation for joint step - please pass one of " "[relu, sigmoid, tanh]")

        activation = activation.lower()

        if activation == 'relu':
            activation = torch.nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()

        layers = (
            [activation]
            + ([torch.nn.Dropout(p=dropout)] if dropout else [])
            + [torch.nn.Linear(joint_n_hidden, num_classes)]
        )
        return pred, enc, torch.nn.Sequential(*layers)

    def joint(self, f: torch.Tensor, g: torch.Tensor):
        """
        Compute the joint step of the network.

        Here,
        B = Batch size
        T = Acoustic model timesteps
        U = Target sequence length
        H1, H2 = Hidden dimensions of the Encoder / Decoder respectively
        H = Hidden dimension of the Joint hidden step.
        V = Vocabulary size of the Decoder (excluding the RNNT blank token).

        NOTE:
            The implementation of this model is slightly modified from the original paper.
            The original paper proposes the following steps :
            (enc, dec) -> Expand + Concat + Sum [B, T, U, H1+H2] -> Forward through joint hidden [B, T, U, H] -- *1
            *1 -> Forward through joint final [B, T, U, V + 1].

            We instead split the joint hidden into joint_hidden_enc and joint_hidden_dec and act as follows:
            enc -> Forward through joint_hidden_enc -> Expand [B, T, 1, H] -- *1
            dec -> Forward through joint_hidden_dec -> Expand [B, 1, U, H] -- *2
            (*1, *2) -> Sum [B, T, U, H] -> Forward through joint final [B, T, U, V + 1].

        Args:
            f: Output of the Encoder model. A torch.Tensor of shape [B, T, H1]
            g: Output of the Decoder model. A torch.Tensor of shape [B, U, H2]

        Returns:
            Logits / log softmaxed tensor of shape (B, T, U, V + 1).
        """
        # f = [B, T, H1]
        f = self.enc(f)
        f.unsqueeze_(dim=2)  # (B, T, 1, H)

        # g = [B, U, H2]
        g = self.pred(g)
        g.unsqueeze_(dim=1)  # (B, 1, U, H)

        inp = f + g  # [B, T, U, H]

        del f, g

        # Forward adapter modules on joint hidden
        # if self.is_adapter_available():
        #     inp = self.forward_enabled_adapters(inp)

        res = self.joint_net(inp)  # [B, T, U, V + 1]

        del inp

        if self.preserve_memory:
            torch.cuda.empty_cache()

        # currently no support for log_softmax
        # TODO: potential support for temperature
        return res

    @typecheck()
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
    ) -> Union[torch.Tensor, List[Optional[torch.Tensor]]]:
        # encoder = (B, D, T)
        # decoder = (B, D, U) if passed, else None
        encoder_outputs = encoder_outputs.transpose(1, 2)  # (B, T, D)
        decoder_outputs = decoder_outputs.transpose(1, 2)  # (B, U, D)

        batch_size = int(encoder_outputs.shape[0])  # actual batch size

        # If fused joint step is required, fused batch size is required as well
        if self._fused_batch_size is None:
            logging.warning("`fused_batch_size` recommended to be set")
            fused_batch_size = batch_size
        else:
            fused_batch_size = self._fused_batch_size

        # When using fused joint step, both encoder and transcript lengths must be provided
        if (encoder_lengths is None) or (targets_lengths is None):
            raise ValueError("encoder and target lengths must be provided")

        losses = []
        target_lengths = []

        # Iterate over batch using fused_batch_size steps
        for batch_idx in range(0, batch_size, fused_batch_size):
            begin = batch_idx
            end = min(begin + fused_batch_size, batch_size)

            # Extract the sub batch inputs
            # sub_enc = encoder_outputs[begin:end, ...]
            # sub_targets = targets[begin:end, ...]
            sub_enc = encoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))
            sub_targets = targets.narrow(dim=0, start=begin, length=int(end - begin))

            sub_enc_lens = encoder_lengths[begin:end]
            sub_targets_lens = targets_lengths[begin:end]

            # Sub targetss does not need the full padding of the entire batch
            # Therefore reduce the decoder time steps to match
            max_sub_enc_length = sub_enc_lens.max()
            max_sub_targets_length = sub_targets_lens.max()

            # Reduce encoder length to preserve computation
            # Encoder: [sub-batch, T, D] -> [sub-batch, T', D]; T' < T
            if sub_enc.shape[1] != max_sub_enc_length:
                sub_enc = sub_enc.narrow(dim=1, start=0, length=int(max_sub_enc_length))

            # sub_dec = decoder_outputs[begin:end, ...]  # [sub-batch, U, D]
            sub_dec = decoder_outputs.narrow(dim=0, start=begin, length=int(end - begin))  # [sub-batch, U, D]

            # Reduce decoder length to preserve computation
            # Decoder: [sub-batch, U, D] -> [sub-batch, U', D]; U' < U
            if sub_dec.shape[1] != max_sub_targets_length + 1:
                sub_dec = sub_dec.narrow(dim=1, start=0, length=int(max_sub_targets_length + 1))

            # Perform joint => [sub-batch, T', U', V + 1]
            sub_joint = self.joint(sub_enc, sub_dec)

            del sub_dec

            # Reduce targets length to correct alignment
            # targets: [sub-batch, L] -> [sub-batch, L']; L' <= L
            if sub_targets.shape[1] != max_sub_targets_length:
                sub_targets = sub_targets.narrow(dim=1, start=0, length=int(max_sub_targets_length))

            # Compute sub batch loss
            # preserve loss reduction type
            loss_reduction = self.loss.reduction

            # override loss reduction to sum
            self.loss.reduction = None

            # compute and preserve loss
            loss_batch = self.loss(
                blank_logits=sub_joint[..., 0],
                predictions=sub_joint[..., 1:],
                targets=sub_targets,
                encoder_lengths=sub_enc_lens,
                target_lengths=sub_targets_lens,
            )
            losses.append(loss_batch)
            target_lengths.append(sub_targets_lens)

            # reset loss reduction type
            self.loss.reduction = loss_reduction

            # # Update WER for sub batch
            # if compute_wer:
            #     sub_enc = sub_enc.transpose(1, 2)  # [B, T, D] -> [B, D, T]
            #     sub_enc = sub_enc.detach()
            #     sub_targetss = sub_targets.detach()
            #
            #     # Update WER on each process without syncing
            #     self.wer.update(sub_enc, sub_enc_lens, sub_targetss, sub_targets_lens)
            # TODO: implement MSE

            del sub_enc, sub_targets, sub_enc_lens, sub_targets_lens

        # Reduce over sub batches
        if losses is not None:
            losses = self.loss.reduce(losses, target_lengths)

        return losses

    @property
    def num_classes_with_blank(self):
        return 2
