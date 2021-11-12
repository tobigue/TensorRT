#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#

import argparse

from collections import namedtuple, OrderedDict
from itertools import product
from typing import Dict

# TRT-HuggingFace
from NNDF.networks import Precision, NetworkMetadata, NNConfig, Dims
from NNDF.interface import MetadataArgparseInteropMixin

# Limitation of namedtuples. You must declare namedtuples in module scope and not in classes.
# Otherwise pickle doesn't work.
# See: https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
_MT5Metadata = namedtuple("MT5Metadata", ["kv_cache"])


class MT5Metadata(_MT5Metadata, MetadataArgparseInteropMixin):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add commandline interface parser."""
        network_group = parser.add_argument_group("MT5 network")
        network_group.add_argument(
            "--variant",
            help="MT5 variant to generate",
            choices=MT5ModelTRTConfig.TARGET_MODELS,
            required=True,
        )
        network_group.add_argument(
            "--enable-kv-cache",
            help="MT5 enable KV cache",
            action="store_true",
            default=False,
        )

    @staticmethod
    def from_args(args: argparse.Namespace):
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=MT5Metadata(kv_cache=args.enable_kv_cache),
        )

    @staticmethod
    def add_inference_args(parser: argparse.ArgumentParser) -> None:
        MT5Metadata.add_args(parser)
        inference_group = parser.add_argument_group("inference group")
        inference_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )

    @staticmethod
    def from_inference_args(args: argparse.Namespace):
        base_metadata = MT5Metadata.from_args(args)
        return base_metadata._replace(precision=Precision(fp16=args.fp16))


class MT5ModelTRTConfig(NNConfig):

    TARGET_MODELS = ["google/mt5-small", "google/mt5-base"]
    NUMBER_OF_LAYERS = {TARGET_MODELS[0]: 8, TARGET_MODELS[1]: 12}
    MAX_SEQUENCE_LENGTH = {
        TARGET_MODELS[0]: 512,
        TARGET_MODELS[1]: 768,
    }

    NETWORK_FULL_NAME = "full"
    NETWORK_DECODER_SEGMENT_NAME = "decoder"
    NETWORK_ENCODER_SEGMENT_NAME = "encoder"
    NETWORK_SEGMENTS = [NETWORK_DECODER_SEGMENT_NAME, NETWORK_ENCODER_SEGMENT_NAME]

    def __init__(self):
        precision_fp16 = [False, True]
        kv_caches = [False, True]

        variants = []
        for variant, fp16, kv_cache in product(
            MT5ModelTRTConfig.TARGET_MODELS, precision_fp16, kv_caches
        ):
            variants.append(
                NetworkMetadata(
                    variant=variant,
                    precision=Precision(fp16=fp16),
                    other=MT5Metadata(kv_cache=kv_cache),
                )
            )

        super().__init__("MT5", variants=variants)

    def get_python_requirements(self):
        base_requirements = super().get_python_requirements()
        base_requirements.append("transformers==4.6.1")
        return base_requirements

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """
        return MT5ModelTRTConfig.NETWORK_SEGMENTS

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        # Remove redundant t5 name
        metadata = metadata._replace(variant=metadata.variant.lstrip("google/mt5-"))
        return super().get_metadata_string(metadata)

    @staticmethod
    def get_input_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of input dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_inputs = Dims(
            OrderedDict(
                {
                    "input_ids": (Dims.BATCH, Dims.SEQUENCE),
                    "encoder_hidden_states": (
                        Dims.BATCH,
                        Dims.create_new_sequence_dim("encoder_hidden_length"),
                        MT5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                    ),
                }
            )
        )

        encoder_inputs = Dims(OrderedDict({"input_ids": (Dims.BATCH, Dims.SEQUENCE)}))

        return {
            MT5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_inputs,
            MT5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_inputs,
        }

    @staticmethod
    def get_output_dims(metadata) -> Dict:
        """
        Returns dictionary encoding of output dimensions.
        Keys will be equal to get_model_segments()

        Returns:
            (Dict[str, Dims]): {"decoder": Dims, "encoder": Dims}
        """
        decoder_outputs = Dims(
            OrderedDict({"hidden_states": (Dims.BATCH, Dims.SEQUENCE)})
        )
        encoder_outputs = Dims(
            OrderedDict(
                {
                    "hidden_states": (
                        Dims.BATCH,
                        Dims.SEQUENCE,
                        MT5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
                    )
                }
            )
        )

        return {
            MT5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME: decoder_outputs,
            MT5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME: encoder_outputs,
        }
