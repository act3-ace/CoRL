"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
The following module contains the implementation for the  stacked observation
"""
import gymnasium
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import Dict, List, TensorType

tf1, tf, tfv = try_import_tf()


class FrameStackingModel(TFModelV2):
    """The following class is a slight modification of the base Fully Connected Model within RLLIB. The model adds on
        the ability to do frame stacking. This is done within the model to (1) enable the HPARAM search over the setting
        for the number of frames - Not possible with environment wrappers (2) enable a more in line view of the
        framestacking inline with paper representation.

        Note: Unlike the environment wrapper which just concatenates the data into one large vector and sends through
        network. This implementation will process N identical obs (vectors) through all hidden layers until the last
        FC layer which inputs are flattened to ensure that we are only producing the expected number of output actions

        The architecture produced is located below:
            - Note: Parallel paths. These are only generated if the Value function is defined as not sharing network.
                    In the case that the network is shared only a single path would exist.
            - Note: All of the default capability of the default RLLIB model is maintained and may not be shown in
                    diagram. Further testing needed for other paths.

                       FC Layers 1-N
                                        |-----|
                            |---------->|Dense|-|
                            |           |-----| |-|        |-------|     |-----|
                            |             |-----| |------->|Flatten|---->| FC  |---|
                            |               |-----|        |-------|     |-----|   |
        N Obs          |--------|                                                  |        Model
        -------------> |        |                                                  |
        (opt) N Reward |        |                                                  |------->[FC Out   ] ------> Actions
        -------------> | Inputs |
        (opt) N Action |        |                                                  |------->[Value Out] ------> Values
        -------------> |--------|                                                  |
                            |           |-----|                                    |
                            |---------->|Dense|-|                                  |
                                        |-----| |-|        |-------|     |-----|   |
                                          |-----| |------->|Flatten|---->| FC  |---|
                                            |-----|        |-------|     |-----|

        |----------------------------------------- |
        | Note: Ensure final output is in          |
        | 1 X ACTION not 1 X Frame X ACTION        |
        |------------------------------------------|

        Inputs  = 1 X Frames X (Obs + Rewards + Actions)
        Outputs = 1 X Actions

        The following is a example summary of the model from tensor flow on the  base Single Environments
        Model: "functional_1"
        __________________________________________________________________________________________________
        Layer (type)                    Output Shape         Param #     Connected to
        ==================================================================================================
        observations (InputLayer)       [(None, 5, 77)]      0
        __________________________________________________________________________________________________
        fc_0 (Dense)                    (None, 5, 256)       19968       observations[0][0]
        __________________________________________________________________________________________________
        fc_value_0 (Dense)              (None, 5, 256)       19968       observations[0][0]
        __________________________________________________________________________________________________
        fc_flatten_1 (Flatten)          (None, 1280)         0           fc_0[0][0]
        __________________________________________________________________________________________________
        fc_value_flatten_1 (Flatten)    (None, 1280)         0           fc_value_0[0][0]
        __________________________________________________________________________________________________
        fc_1 (Dense)                    (None, 256)          327936      fc_flatten_1[0][0]
        __________________________________________________________________________________________________
        fc_value_1 (Dense)              (None, 256)          327936      fc_value_flatten_1[0][0]
        __________________________________________________________________________________________________
        fc_out (Dense)                  (None, 51)           13107       fc_1[0][0]
        __________________________________________________________________________________________________
        value_out (Dense)               (None, 1)            257         fc_value_1[0][0]
        ==================================================================================================
        Total params: 709,172
        Trainable params: 709,172
        Non-trainable params: 0
        __________________________________________________________________________________________________

    Arguments:
        TFModelV2: [description]
    """

    PREV_N_OBS = "prev_n_obs"
    PREV_N_REWARDS = "prev_n_rewards"
    PREV_N_ACTIONS = "prev_n_actions"

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        post_fcnet_hiddens=None,
        num_frames: int = 4,
        include_actions: bool = True,
        include_rewards: bool = True,
    ):
        """Class constructor

        Arguments:
            obs_space (gymnasium.spaces.Space): Observation space of the target gymnasium
                env. This may have an `original_space` attribute that
                specifies how to unflatten the tensor into a ragged tensor.
            action_space (gymnasium.spaces.Space): Action space of the target gymnasium
                env.
            num_outputs (int): Number of output units of the model.
            model_config (ModelConfigDict): Config for the model, documented
                in ModelCatalog.
            name (str): Name (scope) for the model.

        This method should create any variables used by the model.

        Keyword Arguments:
            num_frames {int} -- The number of frames to stack (default: {4})
            include_actions {int} -- Whether or not to include actions as part of frame stacking (default: True)
            include_actions {int} -- Whether or not to include actions as part of frame stacking (default: True)

        Returns:
            [type] -- [description]
        """
        if post_fcnet_hiddens is None:
            post_fcnet_hiddens = []
        # Initializes a ModelV2 object.
        TFModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # This model specific items
        self.num_frames = num_frames

        # Base model items
        self.num_outputs = num_outputs

        # Read out the model configuration parameters passed by RLLIB. Note this is maintained to ensure
        # compatibility with existing setup
        free_log_std, hiddens, activation, no_final_linear, vf_share_layers = self.get_config_opts()

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, ("num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable([0.0] * num_outputs, dtype=tf.float32, name="log_std")

        # Create the input layers for the observations, actions, and rewards
        flattened_action_space = flatten_space(action_space)
        observations, actions, rewards = self.create_input_layers(obs_space, flattened_action_space)

        # Select the input layer configuration based on input arguments
        self.include_rewards = include_rewards
        self.include_actions = include_actions
        self.input_list, self.inputs = FrameStackingModel.select_input_layer_configuration(
            include_rewards, include_actions, observations, actions, rewards
        )

        # Create layers 0 to second-last.
        last_layer = self.create_dense_hidden_layers(hiddens, self.inputs, activation, "fc")

        # The action distribution outputs.
        logits_out, last_layer = self.create_last_fc_layer_output(
            no_final_linear, num_outputs, activation, last_layer, hiddens, post_fcnet_hiddens, obs_space
        )

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                return tf.tile(tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(self.inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)([logits_out, log_std_out])

        last_vf_layer = self.build_vf_network(vf_share_layers, self.inputs, hiddens, post_fcnet_hiddens, activation)

        value_out = tf.keras.layers.Dense(1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01))(
            last_vf_layer if last_vf_layer is not None else last_layer
        )

        self.base_model = tf.keras.Model(self.input_list, [(logits_out if logits_out is not None else last_layer), value_out])
        # print(self.base_model.summary())

        self.register_view_requirements(num_frames, obs_space, flattened_action_space)

        self._value_out = None

    @staticmethod
    def select_input_layer_configuration(include_rewards, include_actions, observations, actions, rewards):
        """Sets up the input layer based on the configuration of the arguments to the model

        Arguments:
            include_rewards {bool} -- Flag to indicate that rewards should be part of frame stacking
            include_actions {bool} -- Flag to indicate that the actions should be part of frame stacking
            observations {Tensor} -- [description]
            actions {Tensor} -- [description]
            rewards {Tensor} -- [description]
        """
        # Last hidden layer output (before logits outputs).
        if include_actions and not include_rewards:
            input_list = [observations, actions]
            inputs = tf.keras.layers.Concatenate(axis=-1)(input_list)
        elif not include_actions and include_rewards:
            input_list = [observations, rewards]
            inputs = tf.keras.layers.Concatenate(axis=-1)(input_list)
        elif include_actions and include_rewards:
            input_list = [observations, actions, rewards]
            inputs = tf.keras.layers.Concatenate(axis=-1)(input_list)
        else:
            input_list = [observations]
            inputs = observations
        return input_list, inputs

    def create_input_layers(self, obs_space, action_space):
        """Creates the input layers for starting the graph

        Arguments:
            obs_space {gymnasium.Space} -- The input space - flattended
            action_space {gymnasium.Space} -- The input space - flattended

        Returns:
            tuple[tensor] -- The input layers for observations, rewards, actions
        """
        # (?, Number of Frames, 1)
        rewards = tf.keras.layers.Input(shape=(self.num_frames, 1), name="rewards")
        # (?, Number of Frames, len obs flatten)
        observations = tf.keras.layers.Input(shape=(self.num_frames, obs_space.shape[0]), name="observations")
        # (?, Number of Frames, len actions flatten)
        actions = tf.keras.layers.Input(shape=(self.num_frames, len(action_space)), name="actions")
        return observations, actions, rewards

    def create_last_fc_layer_output(self, no_final_linear, num_outputs, activation, last_layer, hiddens, post_fcnet_hiddens, obs_space):
        """[summary]

        Arguments:
            no_final_linear: [description]
            num_outputs: [description]
            activation: [description]
            last_layer: [description]
            hiddens: [description]
            obs_space: [description]

        Returns:
            [type] -- [description]
        """
        # The action distribution outputs.
        logits_out = None
        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs, name="fc_out", activation=activation, kernel_initializer=normc_initializer(1.0)
            )(last_layer)

        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                last_layer = FrameStackingModel.flatten_plus_dense(
                    hiddens, post_fcnet_hiddens, last_layer, activation, "fc", len(hiddens) - 1
                )

            if num_outputs:
                logits_out = tf.keras.layers.Dense(num_outputs, name="fc_out", activation=None, kernel_initializer=normc_initializer(0.01))(
                    last_layer
                )
            # Adjust num_outputs to be the number of nodes in the last layer.
            else:
                self.num_outputs = ([int(np.prod(obs_space.shape))] + hiddens[-1:])[-1]
        return logits_out, last_layer

    def build_vf_network(self, vf_share_layers, inputs, hiddens, flatten_plus_dense, activation):
        """Creates the value function network if configured in model config

        Arguments:
            vf_share_layers: [description]
            inputs: [description]
            hiddens: [description]
            activation: [description]

        Returns:
            [type] -- [description]
        """
        last_vf_layer = None
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            value_function_prefix = "fc_value"
            last_vf_layer = self.create_dense_hidden_layers(hiddens, inputs, activation, value_function_prefix)
            last_vf_layer = self.flatten_plus_dense(
                hiddens, flatten_plus_dense, last_vf_layer, activation, value_function_prefix, len(hiddens) - 1
            )
        return last_vf_layer

    def register_view_requirements(self, num_frames: int, obs_space, flattened_action_space):
        """Sets up the view requirements for the forward pass call

        Arguments:
            num_frames {int} -- The number of frames to stack
            obs_space: The observation space definition
            flattened_action_space: flattened action space
        """
        self.view_requirements[FrameStackingModel.PREV_N_OBS] = ViewRequirement(
            data_col="obs", shift=f"-{num_frames - 1}:0", space=obs_space
        )
        if self.include_rewards:
            self.view_requirements[FrameStackingModel.PREV_N_REWARDS] = ViewRequirement(data_col="rewards", shift=f"-{self.num_frames}:-1")
        if self.include_actions:
            self.view_requirements[FrameStackingModel.PREV_N_ACTIONS] = ViewRequirement(
                data_col="actions",
                shift=f"-{self.num_frames}:-1",
                space=gymnasium.spaces.box.Box(low=-np.inf, high=np.inf, shape=(len(flattened_action_space),), dtype=np.int64),
            )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):  # type: ignore
        """Call the model with the given input tensors and state.

        Any complex observations (dicts, tuples, etc.) will be unpacked by __call__ before being passed to forward(). To access
        the flattened observation tensor, refer to input_dict[“obs”].

        This method can be called any number of times. In eager execution, each call to forward() will eagerly evaluate the model.
        In symbolic execution, each call to forward creates a computation graph that operates over the variables of this model
        (i.e., shares weights).

        Custom models should override this instead of __call__.

        Arguments:
            input_dict (dict) - dictionary of input tensors, including “obs”, “obs_flat”, “prev_action”, “prev_reward”, “is_training”,
                                “eps_id”, “agent_id”, “infos”, and “t”.
            state (list) - list of state tensors with sizes matching those returned by get_initial_state + the batch dimension
            seq_lens (Tensor) - 1d tensor holding input sequence lengths

        Returns:
            The model output tensor of size [BATCH, num_outputs], and the new RNN state.
        """

        if self.include_actions and not self.include_rewards:
            model_out, self._value_out = self.base_model(
                [input_dict[FrameStackingModel.PREV_N_OBS], input_dict[FrameStackingModel.PREV_N_ACTIONS]]
            )
        elif not self.include_actions and self.include_rewards:
            model_out, self._value_out = self.base_model(
                [input_dict[FrameStackingModel.PREV_N_OBS], input_dict[FrameStackingModel.PREV_N_REWARDS]]
            )
        elif self.include_actions and self.include_rewards:
            model_out, self._value_out = self.base_model(
                [
                    input_dict[FrameStackingModel.PREV_N_OBS],
                    input_dict[FrameStackingModel.PREV_N_ACTIONS],
                    input_dict[FrameStackingModel.PREV_N_REWARDS],
                ]
            )
        else:
            model_out, self._value_out = self.base_model([input_dict[FrameStackingModel.PREV_N_OBS]])
        return model_out, state

    def value_function(self) -> TensorType:
        """Returns the value function output for the most recent forward pass.

        Note that a forward call has to be performed first, before this methods can return anything and thus
        that calling this method does not cause an extra forward pass through the network.

        Returns:
            value estimate tensor of shape [BATCH].
        """
        return tf.reshape(self._value_out, [-1])

    def get_config_opts(self):
        """Gets the configuration options utilizes by the frame stacking model

        Returns:
            Tuple -- configuration options for the models (Bool, list[int], function, Bool, Bool)
        """
        hiddens = self.model_config.get("fcnet_hiddens", []) + self.model_config.get("post_fcnet_hiddens", [])
        activation = self.model_config.get("fcnet_activation")
        if not self.model_config.get("fcnet_hiddens", []):
            activation = self.model_config.get("post_fcnet_activation")
        activation = get_activation_fn(activation)
        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")
        free_log_std = self.model_config.get("free_log_std")
        return free_log_std, hiddens, activation, no_final_linear, vf_share_layers

    @staticmethod
    def create_dense_hidden_layers(hiddens, layer, activation, prefix: str):
        """Creates the hidden dense layers

        Arguments:
            hiddens {List[int]} -- The list of hidden layers for the FC components
            layer {Tensor} -- [description] --- TODO Remove as not needed to pass in...
            activation {Function} -- [description]
            prefix {str} -- The string to use for the naming of the layer

        Returns:
            [type] -- [description]
        """
        for index, size in enumerate(hiddens[:-1]):
            dense_name = f"{prefix}_{index}"
            layer = tf.keras.layers.Dense(size, name=dense_name, activation=activation, kernel_initializer=normc_initializer(1.0))(layer)
        return layer

    @staticmethod
    def flatten_plus_dense(hiddens, post_fcnet_hiddens, layer, activation, prefix: str, index: int):
        """Creates the final/last dense layer with flatten to ensure the correct output size

        Arguments:
            hiddens {List[int]} -- List containing the size of each hidden layer
            layer {Tensor} -- [description]
            activation {function]} -- [description]
            prefix {str} -- The string to use when creating the layers
            index {int} -- The index of the layer to add the flatten on.

        Returns:
            Tensor -- The layer just created with flatten + FD
        """
        flatten_name = f"{prefix}_flatten_{index}"
        dense_name = f"{prefix}_{index}"
        layer = tf.keras.layers.Flatten(name=flatten_name)(layer)
        layer = tf.keras.layers.Dense(hiddens[index], name=dense_name, activation=activation, kernel_initializer=normc_initializer(1.0))(
            layer
        )
        for index_cat, size in enumerate(post_fcnet_hiddens):
            layer = tf.keras.layers.Dense(
                size, name=f"{dense_name}_cat_{index_cat}", activation=activation, kernel_initializer=normc_initializer(1.0)
            )(layer)
        return layer


ModelCatalog.register_custom_model("FrameStackingModel", FrameStackingModel)
