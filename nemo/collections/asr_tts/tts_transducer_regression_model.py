# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning import Trainer
from transformers import EncodecModel

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text import AudioTextBatchWithSpeakerId
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset, DALIOutputs
from nemo.collections.asr.losses.rnnt import RNNTLossMse
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER, RNNTDecoding
from nemo.collections.asr.models import EncDecRNNTBPEModel, EncDecRNNTModel
from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint, rnnt
from nemo.collections.asr.modules.audio_preprocessing import AudioPreprocessor
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr_tts.modules.rnnt.joint_regression import FactorizedRegressionJoint
from nemo.collections.asr_tts.modules.rnnt.regression_decoder import TransducerRegressionDecoder
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes import Exportable, ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


class TextToSpeechTransducerRegressionModel(ModelPT, Exportable):
    class DatasetType(PrettyStrEnum):
        ASR_BPE = "asr_bpe"
        TTS = "tts"

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.world_size = 1 if trainer is None else trainer.world_size
        self.dataset_type = self.DatasetType(cfg.dataset_type)  # setup before super

        # Setup the tokenizer
        self.tokenizer: TokenizerSpec

        if self.dataset_type == self.DatasetType.ASR_BPE:
            self._setup_tokenizer(cfg.tokenizer)
            vocabulary = self.tokenizer.tokenizer.get_vocab()
            vocabulary_size = len(vocabulary)
            if cfg.encoder["_target_"] == "nemo.collections.tts.modules.transformer.FFTransformerEncoder":
                cfg.encoder["n_embed"] = vocabulary_size + 1
                cfg.encoder["padding_idx"] = vocabulary_size  # TODO: is this correct?
        elif self.dataset_type == self.DatasetType.TTS:
            raise NotImplementedError("TTS dataset support is WIP")
        else:
            raise NotImplementedError("Unknown dataset type")

        super().__init__(cfg=cfg, trainer=trainer)

        # self.text_embeddings = nn.Embedding(vocabulary_size, cfg.text_embedding_dim)

        # Initialize components
        self.preprocessor: AudioPreprocessor = ModelPT.from_config_dict(self.cfg.preprocessor)
        self.encoder: nn.Module = ModelPT.from_config_dict(self.cfg.encoder)
        self.decoder: TransducerRegressionDecoder = ModelPT.from_config_dict(self.cfg.decoder)
        self.joint: FactorizedRegressionJoint = ModelPT.from_config_dict(self.cfg.joint)
        assert isinstance(self.preprocessor, AudioPreprocessor)
        assert isinstance(self.decoder, TransducerRegressionDecoder)
        assert isinstance(self.joint, FactorizedRegressionJoint)

        # Setup RNNT Loss
        loss_name, loss_kwargs = EncDecRNNTModel.extract_rnnt_loss_cfg(self, self.cfg.get("loss", None))

        self.loss = RNNTLossMse(
            loss_name=loss_name, loss_kwargs=loss_kwargs, reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
        )

    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "transcript": NeuralType(('B', 'T'), LabelsType()),
    #         "transcript_len": NeuralType(tuple('B'), LengthsType()),
    #         "speaker_ids": NeuralType(tuple('B'), LengthsType()),  # TODO: speaker type???
    #     }

    # @typecheck()
    def forward(self, transcript: torch.Tensor, transcript_len: torch.Tensor, speaker_ids: torch.Tensor):
        raise NotImplementedError

    def setup_training_data(self, train_data_config: Union[DictConfig, Dict]):
        # EncDecRNNTBPEModel.setup_training_data()
        pass

    def setup_validation_data(self, train_data_config: Union[DictConfig, Dict]):
        pass

    def setup_test_data(self, test_data_config: Union[DictConfig, Dict]):
        pass

    def _setup_tokenizer(self, tokenizer_cfg: DictConfig):
        return ASRBPEMixin._setup_tokenizer(self, tokenizer_cfg=tokenizer_cfg)

    def _setup_monolingual_tokenizer(self, tokenizer_cfg: DictConfig):
        return ASRBPEMixin._setup_monolingual_tokenizer(self, tokenizer_cfg=tokenizer_cfg)

    def _cleanup_aggregate_config_and_artifacts_if_needed(self):
        return ASRBPEMixin._cleanup_aggregate_config_and_artifacts_if_needed(self)

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        return results
