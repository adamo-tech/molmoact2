import json

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import MolmoAct2PreTrainedModel
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
import numpy
from runtime import model_storage
from safetensors.torch import load_file
import torch


MolmoAct2PreTrainedModel._init_weights = lambda self, module: None

CHECKPOINTS = {
    "allenai/MolmoAct2": "hf://allenai/MolmoAct2@e432d85f6e039edca44afb93c262f3084ab72a9c",
    "allenai/MolmoAct2-BimanualYAM": "hf://allenai/MolmoAct2-BimanualYAM@8dcbed66f2380e4393189c303ea72488eb9e63c2",
    "allenai/MolmoAct2-DROID": "hf://allenai/MolmoAct2-DROID@d8c1abd8a27d8e859455bbe514df2bcc617db0fb",
    "allenai/MolmoAct2-SO100_101": "hf://allenai/MolmoAct2-SO100_101@152569fe57914d97be91055800035f54e250d009",
}
CHECKPOINT_CHECKSUMS = {
    CHECKPOINTS["allenai/MolmoAct2"]: "2a8557bec0012ad7c60d9ef1196229f35f6c38aa968f5df8ae8f771f6a5b96e3",
    CHECKPOINTS["allenai/MolmoAct2-BimanualYAM"]: "9a1d6482a679a7f88af5a5025d688edfb4fa2bc652f748395b45782135ea5e29",
    CHECKPOINTS["allenai/MolmoAct2-DROID"]: "3e5888684c04287a8ea576beafaa6653ce0bb69f497cc89b4a4948d13ca3e5f4",
    CHECKPOINTS["allenai/MolmoAct2-SO100_101"]: "0f8d915e758cdaa34d9ce83f01b6389cc400f1a86d3cf2aef588fb488da73905",
}
NORM_TAGS = {
    "allenai/MolmoAct2-BimanualYAM": "yam_dual_molmoact2",
    "allenai/MolmoAct2-DROID": "franka_droid",
    "allenai/MolmoAct2-SO100_101": "so100_so101_molmoact2",
}
ACTION_HORIZON = 30
MAX_ACTION_SIZE = MolmoAct2Config.expected_max_action_dim
IMAGE_PREFIX = "observation.images."
IMAGE_SIZE = 378
CONTROL_MODE = "absolute joint pose"


def checkpoint(uri):
    return model_storage.fetch(uri, CHECKPOINT_CHECKSUMS[uri])


def columns(episodes, keys):
    return numpy.vstack(
        [
            numpy.concatenate(
                [
                    numpy.asarray(episode[key], dtype=numpy.float32).reshape(len(episode[key]), -1)
                    for key in keys
                ],
                axis=1,
            )
            for episode in episodes
        ]
    )


def summary(values):
    return {
        "min": numpy.min(values, axis=0).tolist(),
        "max": numpy.max(values, axis=0).tolist(),
        "mean": numpy.mean(values, axis=0).tolist(),
        "std": numpy.std(values, axis=0).tolist(),
        "q01": numpy.quantile(values, 0.01, axis=0).tolist(),
        "q99": numpy.quantile(values, 0.99, axis=0).tolist(),
        "mask": [True] * values.shape[1],
    }


def statistics(episodes, mapping):
    return {
        OBS_STATE: summary(columns(episodes, mapping["state"])),
        ACTION: summary(columns(episodes, mapping["action"])),
    }


def tag_metadata(checkpoint_directory, norm_tag):
    return json.loads((checkpoint_directory / "norm_stats.json").read_text())["metadata_by_tag"][norm_tag]


def tag_mapping(mapping, metadata):
    if metadata["camera_keys"] and len(mapping["images"]) != len(metadata["camera_keys"]):
        raise ValueError(
            f"This robot's pretrained cameras are {', '.join(key.removeprefix(IMAGE_PREFIX) for key in metadata['camera_keys'])}, in that order, and the mapping has {len(mapping['images'])} images."
        )

    return {
        **mapping,
        "setup_type": metadata["setup_type"],
        "control_mode": metadata["control_mode"],
        "action_horizon": metadata["action_horizon"],
    }


def tag_statistics(metadata):
    return {
        key: {name: value for name, value in metadata[stats_key].items() if name != "names"}
        for key, stats_key in ((OBS_STATE, "state_stats"), (ACTION, "action_stats"))
    }


def prompt_setup(parts):
    arm_sides = {part["side"] for part in parts if part["kind"] == "joints"}
    return {
        "setup_type": "bimanual robotic arms" if len(arm_sides) == 2 else "single robotic arm",
        "control_mode": CONTROL_MODE,
    }


def policy(checkpoint_directory, mapping, statistics, lora_rank=None, is_training=False):
    image_keys = [IMAGE_PREFIX + key for key in mapping["images"]]
    config = MolmoAct2Config(
        checkpoint_path=str(checkpoint_directory),
        chunk_size=mapping["action_horizon"],
        n_action_steps=mapping["action_horizon"],
        setup_type=mapping["setup_type"],
        control_mode=mapping["control_mode"],
        image_keys=image_keys,
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
                for key in image_keys
            },
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(len(statistics[OBS_STATE]["q01"]),)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(statistics[ACTION]["q01"]),))
        },
        train_mode_vlm="lora" if lora_rank else "fft",
        lora_rank=lora_rank or MolmoAct2Config.lora_rank,
        gradient_checkpointing=is_training,
        device="cuda",
    )
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(config, dataset_stats=statistics)
    return MolmoAct2Policy(config).to("cuda"), preprocessor, postprocessor


@torch.no_grad()
def load_trainable(molmoact2_policy, path):
    trainable = load_file(path, device="cuda")

    for name, parameter in molmoact2_policy.named_parameters():
        if parameter.requires_grad:
            parameter.copy_(trainable[name])


def clipped_passthrough(values, feature_statistics):
    return numpy.where(feature_statistics["mask"], values, numpy.clip(values, -1.0, 1.0))


def batch(images, states, prompts):
    return {
        **{IMAGE_PREFIX + key: numpy.stack(values) for key, values in images.items()},
        OBS_STATE: torch.from_numpy(numpy.stack(states)),
        "task": prompts,
    }
