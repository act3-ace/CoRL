#  type: ignore
#  flake8: noqa
import logging

import gymnasium
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import AppendBiasLayer, SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class TorchFrameStack(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    PREV_N_OBS = "prev_n_obs"

    def __init__(
        self,
        obs_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        num_frames: int = 1,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", []))
        post_fcnet_hiddens = list(model_config.get("post_fcnet_hiddens", []))
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")

        num_frames = model_config["custom_model_config"].get("num_frames", 1)

        self.view_requirements[TorchFrameStack.PREV_N_OBS] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(num_frames - 1), space=obs_space
        )
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, ("num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(obs_space.shape[-1])
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=size, initializer=normc_initializer(1.0), activation_fn=activation))
            prev_layer_size = size

        layers.append(nn.Flatten())
        prev_layer_size = size * num_frames

        for size in post_fcnet_hiddens:
            layers.append(SlimFC(in_size=prev_layer_size, out_size=size, initializer=normc_initializer(1.0), activation_fn=activation))
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(1.0), activation_fn=activation)
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(in_size=prev_layer_size, out_size=hiddens[-1], initializer=normc_initializer(1.0), activation_fn=activation)
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size, out_size=num_outputs, initializer=normc_initializer(0.01), activation_fn=None
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[-1]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(obs_space.shape[-1])
            vf_layers = []
            for size in hiddens[:-1]:
                vf_layers.append(
                    SlimFC(in_size=prev_vf_layer_size, out_size=size, activation_fn=activation, initializer=normc_initializer(1.0))
                )
                prev_vf_layer_size = size
            vf_layers.append(nn.Flatten())
            prev_vf_layer_size = size * num_frames

            vf_layers.append(
                SlimFC(in_size=prev_vf_layer_size, out_size=hiddens[-1], activation_fn=activation, initializer=normc_initializer(1.0))
            )
            prev_vf_layer_size = hiddens[-1]

            # for size in hiddens:
            #     vf_layers.append(
            #         SlimFC(
            #             in_size=prev_vf_layer_size,
            #             out_size=size,
            #             activation_fn=activation,
            #             initializer=normc_initializer(1.0)))
            #     prev_vf_layer_size = size

            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(in_size=prev_layer_size, out_size=1, initializer=normc_initializer(0.01), activation_fn=None)

        # print("*************************************************")
        # print(model_config)
        # print("************************************************")
        # print(num_frames)
        # print("************************************************")
        # print(self._hidden_layers)
        # print(self._logits)
        # print("***********************************************")
        # print(self._value_branch_separate)
        # print(self._value_branch)

        # # exit(1)
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        self._last_flat_in = input_dict[TorchFrameStack.PREV_N_OBS].float()
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(self._value_branch_separate(self._last_flat_in)).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)


ModelCatalog.register_custom_model("TorchFrameStack", TorchFrameStack)
