# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------
"""
The following file contains:
 (1) FillActions class which back updates/corrects the batch observations with correct actions
 (2)
"""

from gymnasium.spaces.utils import flatten_space
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from corl.environment.centralized_critic_environment import CorlCentralizedCriticEnvWrapper

torch, nn = try_import_torch()


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):  # type: ignore
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    OBS_STR = "obs"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        #
        # Initialize the base classes
        #
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        #
        # Create the action model which is used for decentralized execution
        #
        own_obs_box_space = flatten_space(obs_space.original_space[CorlCentralizedCriticEnvWrapper.OWN_OBS_STR])
        self.action_model = TorchFC(
            own_obs_box_space,  # Box representation of the Agent Own Observations
            action_space,  # Action space for the agent
            num_outputs,
            model_config,
            name + "_action",
        )

        #
        # Create the value function model which is used for centralized training
        #
        self.value_model = TorchFC(obs_space, action_space, 1, model_config, name + "_vf")

        #
        # Store the number of own obs for the forward pass processing
        #
        self.input_own_obs_shape = own_obs_box_space.shape[0]

        #
        # Create state to hold the `input_dict` in the `forward` call as it contains
        # all of the information needed for the centralized critic ()
        #
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        """Execution time model training (Decentralized execution) which only runs on the agent obs and not privalaged information"""
        # Store model-input for possible `value_function()` call.
        self._model_in = [input_dict[CorlCentralizedCriticEnvWrapper.OBS_FLAT_STR], state, seq_lens]

        # get the input flattended data
        # read out only the input observations for the agent
        obs = input_dict[CorlCentralizedCriticEnvWrapper.OBS_FLAT_STR].float()
        input_dict[TorchCentralizedCriticModel.OBS_STR][CorlCentralizedCriticEnvWrapper.OWN_OBS_STR][
            CorlCentralizedCriticEnvWrapper.OBS_FLAT_STR
        ] = obs.split(self.input_own_obs_shape, dim=1)[0]
        return self.action_model.forward(
            input_dict[TorchCentralizedCriticModel.OBS_STR][CorlCentralizedCriticEnvWrapper.OWN_OBS_STR], state, seq_lens
        )

    def value_function(self):
        """Centralized Critic which takes in privalaged information for training the value function model (Centralized Trainer)"""
        value_out, _ = self.value_model(
            {TorchCentralizedCriticModel.OBS_STR: self._model_in[0]},
            self._model_in[1],
            self._model_in[2],
        )
        return torch.reshape(value_out, [-1])


ModelCatalog.register_custom_model("TorchCentralizedCriticModel", TorchCentralizedCriticModel)
