import concurrent.futures
import io
import itertools
import json
import math
import random

from lerobot.utils.constants import ACTION, OBS_STATE
import numpy
from peft.helpers import disable_input_dtype_casting
from PIL import Image
from safetensors.torch import save_file
import torch

from adamo_model import model


PREPARE_THREAD_COUNT = 8
CHUNK_SIZE = 32
TRAINABLE_FILE = "trainable.safetensors"
STATISTICS_FILE = "statistics.json"
OPTIMIZER_FILE = "optimizer.pt"
MAPPING_KEYS = ("images", "state", "action", "frame_rate", "state_sizes", "action_sizes", "setup_type", "control_mode", "action_horizon")


def example(datum, frames):
    frame_images = {}
    images = {}

    for key, image in datum["images"].items():
        if isinstance(image, dict):
            pointer = (image["dataset"], image["episode"], image["frame"])
            frame_images[pointer] = frame_images.get(pointer) or frames.frame(*pointer)
            image = frame_images[pointer][key]

        if isinstance(image, bytes):
            image = numpy.asarray(Image.open(io.BytesIO(image)).convert("RGB"))

        images[key] = image

    state = numpy.concatenate(
        [numpy.ravel(numpy.asarray(value, dtype=numpy.float32)) for value in datum["state"].values()]
    )
    actions = numpy.concatenate(
        [numpy.asarray(value, dtype=numpy.float32) for value in datum["loss_fn_inputs"].values()], axis=1
    )
    return images, state, actions, datum["prompt"]


class Trainer:
    base_model = "allenai/MolmoAct2"
    action_horizon = model.ACTION_HORIZON

    def __init__(self, rank, frames, state_directory=None, seed=None, parts=None):
        self.rank = rank
        self.frames = frames
        self.parts = parts
        self.seed = random.randrange(2**31) if seed is None else seed
        self.mapping = None
        self.episodes = []
        self.policy = None
        self.step_count = 0
        self.preparers = concurrent.futures.ThreadPoolExecutor(max_workers=PREPARE_THREAD_COUNT)

        if state_directory:
            saved = json.loads((state_directory / "mapping.json").read_text())
            self.mapping = {name: saved[name] for name in MAPPING_KEYS}
            self.seed = saved["seed"]
            self.step_count = saved["step_count"]
            self.parts = saved["embodiment"]

        torch.manual_seed(self.seed)
        self.checkpoint = model.checkpoint(model.CHECKPOINTS[self.base_model])
        self.metadata = model.tag_metadata(self.checkpoint, model.NORM_TAGS[self.base_model]) if self.base_model in model.NORM_TAGS else None

        if state_directory:
            self.start(json.loads((state_directory / STATISTICS_FILE).read_text()))
            model.load_trainable(self.policy, state_directory / TRAINABLE_FILE)
            self.optimizer.load_state_dict(
                torch.load(state_directory / OPTIMIZER_FILE, map_location="cuda", weights_only=True)
            )

    def add_episodes(self, episodes):
        self.episodes += episodes

    def start(self, statistics):
        self.statistics = statistics
        self.policy, self.preprocessor, _ = model.policy(
            self.checkpoint, self.mapping, statistics, lora_rank=self.rank, is_training=True
        )
        self.policy.train()
        self.params = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr=0.0)

    def run_statistics(self, data):
        if self.metadata:
            return model.tag_statistics(self.metadata)

        if self.episodes:
            return model.statistics(self.episodes, self.mapping)

        image = next(iter(data[0]["images"].values()))

        if not isinstance(image, dict):
            raise ValueError(
                "This run has no normalization stats. Call `training.compute_norm_stats(episodes)` before `forward_backward`."
            )

        return model.statistics(self.frames.episodes(image["dataset"]), self.mapping)

    def check_mapping(self, datum):
        setup = (
            {"setup_type": self.metadata["setup_type"], "control_mode": self.metadata["control_mode"]}
            if self.metadata
            else model.prompt_setup(self.parts)
        )
        mapping = {
            "images": list(datum["images"]),
            "state": list(datum["state"]),
            "action": list(datum["loss_fn_inputs"]),
            "frame_rate": datum["frame_rate"],
            "state_sizes": [numpy.size(value) for value in datum["state"].values()],
            "action_sizes": [len(value[0]) for value in datum["loss_fn_inputs"].values()],
            "action_horizon": self.action_horizon,
            **setup,
        }
        self.mapping = self.mapping or mapping

        if mapping != self.mapping:
            raise ValueError(
                f"This run reads {self.mapping}, and a datum has {mapping}. A run keeps one set of image, state and action keys and one frame rate."
            )

        if self.metadata and (
            sum(mapping["state_sizes"]) != len(self.metadata["state_stats"]["q01"])
            or sum(mapping["action_sizes"]) != len(self.metadata["action_stats"]["q01"])
        ):
            raise ValueError(
                f"{self.base_model} takes {len(self.metadata['state_stats']['q01'])} state and {len(self.metadata['action_stats']['q01'])} action values, and a datum has {sum(mapping['state_sizes'])} and {sum(mapping['action_sizes'])}."
            )

        if sum(mapping["action_sizes"]) > model.MAX_ACTION_SIZE:
            raise ValueError(
                f"A datum has {sum(mapping['action_sizes'])} action values, and {self.base_model} takes at most {model.MAX_ACTION_SIZE}."
            )

        horizon = len(next(iter(datum["loss_fn_inputs"].values())))

        if horizon != self.action_horizon:
            raise ValueError(f"Action chunks have {horizon} steps, and this base model takes {self.action_horizon}.")

    def prepare(self, data):
        for datum in data:
            self.check_mapping(datum)

        if self.policy is None:
            self.start(self.run_statistics(data))

        examples = list(self.preparers.map(example, data, itertools.repeat(self.frames)))
        batches = []

        for index in range(0, len(examples), CHUNK_SIZE):
            images, states, actions, prompts = zip(*examples[index : index + CHUNK_SIZE])
            batch = model.batch(
                {key: [example_images[key] for example_images in images] for key in self.mapping["images"]},
                [model.clipped_passthrough(state, self.statistics[OBS_STATE]) for state in states],
                list(prompts),
            )
            actions = [model.clipped_passthrough(action, self.statistics[ACTION]) for action in actions]
            batches.append(self.preprocessor({**batch, ACTION: torch.from_numpy(numpy.stack(actions))}))

        return batches

    def forward_backward(self, batches, example_count):
        loss = 0.0

        with disable_input_dtype_casting(self.policy.model):
            for batch in batches:
                batch_loss, _ = self.policy.forward(batch)
                weight = batch[OBS_STATE].shape[0] / example_count
                (batch_loss * weight).backward()
                loss += batch_loss.item() * weight

        return {"loss": loss, "metrics": {}}

    def optim_step(self, adam_params):
        for group in self.optimizer.param_groups:
            group.update(
                lr=adam_params["learning_rate"],
                betas=(adam_params["beta1"], adam_params["beta2"]),
                eps=adam_params["eps"],
                weight_decay=adam_params["weight_decay"],
            )

        clip = adam_params["grad_clip_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, clip if clip > 0 else math.inf)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        return {"grad_norm": float(grad_norm), "step": self.step_count}

    def save_weights(self, directory):
        save_file(
            {
                name: parameter.detach().contiguous()
                for name, parameter in self.policy.named_parameters()
                if parameter.requires_grad
            },
            directory / TRAINABLE_FILE,
        )
        (directory / STATISTICS_FILE).write_text(json.dumps(self.statistics))
        mapping = {
            **self.mapping,
            "base_model": self.base_model,
            "rank": self.rank,
            "embodiment": self.parts,
            "step_count": self.step_count,
            "seed": self.seed,
        }
        (directory / "mapping.json").write_text(json.dumps(mapping))
        return mapping

    def save_state(self, directory):
        torch.save(self.optimizer.state_dict(), directory / OPTIMIZER_FILE)
        return self.save_weights(directory)
