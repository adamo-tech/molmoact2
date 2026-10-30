import json

from lerobot.lerobot_types import TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.policies.molmoact2.molmoact2_hf_model import inference
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2MaskedNormalizerProcessorStep
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from runtime import model_storage
import torch

from adamo_model import model
from adamo_model.trainer import STATISTICS_FILE, TRAINABLE_FILE


MAX_GUIDANCE_WEIGHT = RTCConfig.max_guidance_weight


def capture_cuda_graph(function, device, *, after_warmup):
    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))

    with torch.cuda.stream(warmup_stream):
        function()

    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    after_warmup()
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        output = function()

    return graph, output


inference._capture_cuda_graph = capture_cuda_graph


def guidance_weight(t):
    if t == 0:
        return MAX_GUIDANCE_WEIGHT

    return min((1 - t) / t * ((1 - t) ** 2 + t**2) / (1 - t) ** 2, MAX_GUIDANCE_WEIGHT)


class MolmoAct2:
    def __init__(self, policy, preprocessor, postprocessor, statistics, mapping):
        self.policy = policy.eval().requires_grad_(False)

        for module in self.policy.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(torch.bfloat16)

        self.backbone = self.policy._backbone()
        self.action_expert = self.backbone._require_action_expert()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.normalizer = next(
            step for step in preprocessor.steps if isinstance(step, MolmoAct2MaskedNormalizerProcessorStep)
        )
        self.prefix_weights = RTCProcessor(RTCConfig()).get_prefix_weights
        self.state_statistics = statistics[OBS_STATE]
        self.mapping = {name: mapping[name] for name in ("images", "state", "action", "frame_rate")}
        self.action_sizes = mapping.get("action_sizes")
        self.horizon = mapping["action_horizon"]
        self.refill_step_index = self.horizon // 2
        self.graph_key = None

    def normalized(self, actions):
        transition = {TransitionKey.ACTION: torch.from_numpy(actions[None]).to("cuda")}
        return self.normalizer(transition)[TransitionKey.ACTION].clamp(-1.0, 1.0)

    def mask(self, tensor, action_dim_is_pad):
        return self.backbone._mask_action_dim_tensor(
            tensor, action_dim_is_pad=action_dim_is_pad, enabled=self.policy.config.mask_action_dim_padding
        )

    def flow_inputs(self, batch):
        model_inputs = self.policy._model_inputs(batch)
        input_ids, attention_mask = model_inputs.get("input_ids"), model_inputs.get("attention_mask")
        outputs = self.backbone(**model_inputs, use_cache=True, output_attentions=False, output_hidden_states=False)
        encoder_kv_states = self.backbone._extract_kv_states(outputs.past_key_values)
        encoder_attention_mask = self.policy._encoder_attention_mask_for_action_expert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        depth_gate, depth_mask = self.backbone._depth_gate_from_condition(
            input_ids=input_ids, encoder_attention_mask=encoder_attention_mask, layer_kv_states=encoder_kv_states
        )
        encoder_kv_states = self.backbone._apply_depth_gate_to_layer_kv_states(encoder_kv_states, depth_mask, depth_gate)
        noise = torch.randn(1, self.horizon, self.backbone.config.max_action_dim, device="cuda")
        trajectory = self.mask(noise, batch["action_dim_is_pad"])
        context = self.action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=None,
            batch_size=1,
            seq_len=self.horizon,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        steps = int(self.backbone.config.flow_matching_num_steps)
        timesteps = [torch.full((1,), index / steps, device="cuda") for index in range(steps)]
        modulations = self.action_expert.get_or_prepare_modulation_cache(
            timesteps, cache_key=(steps, 1, trajectory.device, trajectory.dtype)
        )
        return inference._ActionFlowInputs(trajectory, context, modulations, batch["action_dim_is_pad"]), steps

    def guided_flow(self, inputs, steps, prefix, weights):
        trajectory = inputs.trajectory

        for index in range(steps):
            t = index / steps
            modulation = inputs.modulations[index]
            velocity = self.mask(
                self.action_expert.forward_with_context(
                    trajectory, modulation.conditioning, context=inputs.context, modulation=modulation
                ),
                inputs.action_dim_is_pad,
            )
            correction = (prefix - trajectory - (1 - t) * velocity) * weights
            trajectory = self.mask(trajectory + (velocity + guidance_weight(t) * correction) / steps, inputs.action_dim_is_pad)

        return trajectory

    def replay_guided_flow(self, inputs, steps, prefix, weights):
        key = inference._cuda_graph_key(inputs, steps)

        if key != self.graph_key:
            self.graph_inputs = inference._clone_static_inputs(inputs)
            self.graph_prefix = prefix.clone()
            self.graph_weights = weights.clone()
            self.graph, self.graph_output = capture_cuda_graph(
                lambda: self.guided_flow(self.graph_inputs, steps, self.graph_prefix, self.graph_weights),
                inputs.trajectory.device,
                after_warmup=lambda: self.graph_inputs.trajectory.copy_(inputs.trajectory),
            )
            self.graph_key = key

        inference._copy_inputs_(self.graph_inputs, inputs)
        self.graph_prefix.copy_(prefix)
        self.graph_weights.copy_(weights)
        self.graph.replay()
        return self.graph_output.clone()

    def guided_actions(self, batch, prefix, inference_delay):
        action_size = self.policy._output_action_dim(batch)

        with torch.no_grad(), self.policy._autocast_context():
            inputs, steps = self.flow_inputs(batch)
            guided_prefix = torch.zeros_like(inputs.trajectory)
            guided_prefix[:, : len(prefix), :action_size] = self.normalized(prefix)
            weights = self.prefix_weights(inference_delay, len(prefix), self.horizon).to("cuda")[None, :, None]
            trajectory = self.replay_guided_flow(inputs, steps, guided_prefix, weights)

        return trajectory[:, :, :action_size].float()

    def infer(self, observation, prompt, prefix=None, inference_delay=0, real_time_chunking=True):
        batch = self.preprocessor(
            model.batch(
                {key: [image] for key, image in observation["images"].items()},
                [model.clipped_passthrough(observation["state"], self.state_statistics)],
                [prompt],
            )
        )
        actions = (
            self.guided_actions(batch, prefix, inference_delay)
            if real_time_chunking and prefix is not None
            else self.policy.predict_action_chunk(batch)
        )
        robot_actions = self.postprocessor(actions)[0].numpy()

        if real_time_chunking and self.graph_key is None:
            self.guided_actions(batch, robot_actions[self.refill_step_index :], 0)

        return robot_actions


def load(policy):
    if "model_weights" in policy:
        weights = model_storage.fetch(policy["model_weights"])
        mapping = json.loads((weights / "mapping.json").read_text())
        statistics = json.loads((weights / STATISTICS_FILE).read_text())
        molmoact2_policy, preprocessor, postprocessor = model.policy(
            model.checkpoint(model.CHECKPOINTS[mapping["base_model"]]), mapping, statistics, lora_rank=mapping["rank"]
        )
        model.load_trainable(molmoact2_policy, weights / TRAINABLE_FILE)
        molmoact2_policy.model = molmoact2_policy.model.merge_and_unload()
        return MolmoAct2(molmoact2_policy, preprocessor, postprocessor, statistics, mapping)

    checkpoint = model.checkpoint(policy["checkpoint"])
    metadata = model.tag_metadata(checkpoint, model.NORM_TAGS[policy["base_model"]])
    mapping = model.tag_mapping(policy["mapping"], metadata)
    statistics = model.tag_statistics(metadata)
    return MolmoAct2(*model.policy(checkpoint, mapping, statistics), statistics, mapping)
