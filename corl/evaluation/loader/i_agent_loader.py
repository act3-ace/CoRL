"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Abstract class to load agents
"""
import re
import typing
from abc import abstractmethod
from collections import OrderedDict

from ray.rllib.algorithms import Algorithm


class IAgentLoader:
    """Abstract class to load agents
    """

    @abstractmethod
    def apply_to_algorithm(self, algorithm: Algorithm, policy_to_apply: str) -> None:
        """Applies the agent to a RLLIB algorithm

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply agent to
            policy_to_apply {str} -- Policy in the algorithm to apply weights to
        """

    @property
    def env_config(self) -> typing.Optional[typing.Dict]:
        """Returns the env config used for generating this checkpoint
        Failing to return an env_config indicates that this is non-trainable

        Returns
        -------
        env_config
        """
        return None

    @property
    def agent_id(self) -> typing.Optional[str]:
        """Returns the agent_id (during training) of the checkpoint
        Failing to return an agent_id indicates that this is non-trainable
        Returns
        -------
        agent_id: typing.Optional[str]
        """
        return None

    @staticmethod
    def _apply_policy_to_weights(algorithm: Algorithm, weights: typing.OrderedDict, policy_to_apply: str):
        """Apply a set of weights intended for a given policy to an algorithm

        This method will attempt to resolve weight naming mismatches between pytorch and tensorflow
        Millage may vary

        Arguments:
            algorithm {Algorithm} -- Algorithm to apply weights to
            weights {typing.OrderedDict} -- Given weights to insert into algorithm
            policy_to_apply {str} -- Policy_id in algorithm to apply weights to

        Raises:
            RuntimeError: Thrown if the given weight doesn't match a weight in the architecture
        """

        # Determine if the architecture is expecting a policy name in the weights
        first_expected_weight = list(algorithm.workers.local_worker().get_weights()[policy_to_apply].keys())[0]  # type: ignore
        arch_policy_id_prefix = policy_to_apply == first_expected_weight[0:len(policy_to_apply)]

        # determine the architecture's delimeter
        arch_delimeter = None
        if "/" in first_expected_weight:
            arch_delimeter = "/"
        elif "." in first_expected_weight:
            arch_delimeter = "."

        # Sanity check
        if arch_policy_id_prefix is True and arch_delimeter is None:
            raise RuntimeError("The architecture expects a policy, but there is not a recognized delimeter")

        # Iterate over all weights
        weights_renamed = OrderedDict()
        for key, value in weights.items():

            # Tensorflow/pytorch compatibility
            updated_key = key

            # Look for a prefix of either {policy_id} or a <team><num>
            # Due to pytorch/tensorflow the prefix may or may not exists
            # This code assumes if there is a prefix it will be one of two things:
            #  - "{policy_id}" as set by Training Algorithm (LP)
            #  - <team_name><num> as set by tensor flow checkpoints (I think?)
            #  - Nothing as set by pytorch
            #
            # If the prefix is found remove it and the following character (delimiter)
            # A prefix will be potentially be added back in the future according to architecture
            prefix_to_remove_regex = rf"^(?:{policy_to_apply}|(?:(?:red|blue|green|grey)\d+))"

            m = re.search(prefix_to_remove_regex, updated_key)
            if m is not None:
                updated_key = updated_key[len(m.group(0)) + 1:]

            # If we know the architecture is expecting a delimenter then convert the
            # given weights key to that delimeter
            if arch_delimeter is not None:
                if arch_delimeter == ".":
                    if "/" in updated_key:
                        updated_key = updated_key.replace("/", ".")
                elif arch_delimeter == "/":
                    if "." in updated_key:
                        updated_key = updated_key.replace(".", "/")
                else:
                    raise RuntimeError(f"Unknown delimiter encountered: {arch_delimeter}")

            # If expecting the policy id append
            if arch_policy_id_prefix:
                updated_key = f"{policy_to_apply}{arch_delimeter}{updated_key}"

            assert (
                updated_key in algorithm.workers.local_worker().get_weights()[policy_to_apply].keys()  # type: ignore
            ), f"The {updated_key} weight does not match a known weight in the architecture!"

            weights_renamed[updated_key] = value

        assert (
            weights_renamed.keys()
            == algorithm.workers.local_worker().get_weights()[policy_to_apply].keys()  # type: ignore
        ), "The model used by rllib config and the model that weights are being loaded from don't match in architecture!"

        algorithm.workers.local_worker().set_weights({policy_to_apply: weights_renamed})  # type: ignore
