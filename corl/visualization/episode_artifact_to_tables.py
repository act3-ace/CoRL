"""
---------------------------------------------------------------------------


Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Converts the episode artifact into flattened tables
"""

import copy
import time
import typing
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import flatten_dict
import gymnasium
import numpy as np
import pandas as pd
from pydantic import BaseModel
from ray.rllib.utils.spaces.repeated import Repeated
from tqdm import tqdm

from corl.evaluation.episode_artifact import EpisodeArtifact
from corl.evaluation.evaluation_outcome import EvaluationOutcome
from corl.evaluation.recording.folder import FolderRecord
from corl.libraries.units import Quantity
from corl.visualization.network_explainability.env_policy_transforms import _to_flattened_dict


# The dataclasses act as schemas for the dataframes
@dataclass
class AgentDoneStatus:
    """A flattened structure for the episode artifact done status information
    (Win, Loss, Draws) for each agent"""

    test_case: int
    agent_id: str
    done_name: str
    done_status: str | None = None
    """A done status code may not exist if it was not triggered"""


@dataclass
class PlatformDoneStates:
    """A flattened structure for the episode artifact done states (e.g. which
    dones are triggered) for each platform"""

    test_case: int
    platform_name: str
    done_name: str
    done_triggered: bool


@dataclass
class EpisodeMetaData:
    """A flattened structure for the episode artifact metadata"""

    test_case: int
    worker_index: int
    wall_time_sec: float
    frame_rate: float | None
    steps: int
    seconds: float


@dataclass
class FlattenedAgentStep:
    """A flattened structure for the agent steps"""

    test_case: int
    step: int
    """step index"""
    step_rel_to_end: int | None
    """step index relative to the end of the trajectory"""
    time: float | int | None
    """time = step / frame_rate"""
    time_rel_to_end: float | int | None
    """time, relative to the end of the trajectory"""
    agent_id: str
    attribute_descrip: str
    attribute_name: str
    attribute_value: typing.Any
    units: str | None = None

    @dataclass
    class AttributeDescription:
        """A string that describes the attribute"""

        observation: str = "observation"
        action: str = "action"
        reward: str = "reward"
        cumulative_reward: str = "cumulative_reward"


@dataclass
class SpaceData:
    """A flattened structure containing space definitions and units"""

    agent_id: str
    attribute_name: str
    space_type: str
    min: int | float  # noqa: A003
    max: int | float  # noqa: A003
    normalized_min: int | float
    normalized_max: int | float
    units: str | None = "dimensionless"


class EpisodeArtifactTablesValidator(BaseModel):
    """Validator for EpisodeArtifactTables"""

    input_path: Path
    file_loader: typing.Callable


class EpisodeArtifactTables:
    """Creates flattened tables (e.g. dataframes) for the attributes within the EpisodeArtifact"""

    # reducer param for flattened_dict
    _reducer = "dot"
    _reducer_map = {"dot": "."}

    def __init__(
        self,
        input_path: str | Path,
        file_loader: typing.Callable = FolderRecord,
        status_bar: typing.Callable = tqdm,
        print_func: typing.Callable = print,
    ) -> None:
        input_path = Path(input_path)
        self.config: EpisodeArtifactTablesValidator = EpisodeArtifactTablesValidator(input_path=input_path, file_loader=file_loader)
        self.reader = self.config.file_loader(input_path)
        self.evaluation_outcome: EvaluationOutcome = self.load_episode_artifacts()
        self.status_bar = status_bar
        self.print_func = print_func

        # Contains the reset states for the simulator
        self.initial_states_table = pd.DataFrame(self.evaluation_outcome.test_cases).reset_index(drop=False)

        self.agent_dones_table = pd.DataFrame([], columns=list(AgentDoneStatus.__dataclass_fields__.keys()))
        self.platform_dones_table = pd.DataFrame([], columns=list(PlatformDoneStates.__dataclass_fields__.keys()))
        self.episode_metadata_table = pd.DataFrame([], columns=list(EpisodeMetaData.__dataclass_fields__.keys()))
        self.agent_steps_table = pd.DataFrame([], columns=list(FlattenedAgentStep.__dataclass_fields__.keys()))
        self.space_data_table = pd.DataFrame([], columns=list(SpaceData.__dataclass_fields__.keys()))

        # Populates the tables defined above
        self._populate_tables()

        self.tables = {
            "InitialStates": self.initial_states_table,
            EpisodeMetaData.__name__: self.episode_metadata_table,
            AgentDoneStatus.__name__: self.agent_dones_table,
            PlatformDoneStates.__name__: self.platform_dones_table,
            FlattenedAgentStep.__name__: self.agent_steps_table,
            SpaceData.__name__: self.space_data_table,
        }

    def load_episode_artifacts(self) -> EvaluationOutcome:
        """Calls the reader to load episode artifact"""
        return self.reader.load()

    def _populate_tables(self):
        episode_metadata = []
        agent_done_statuses = []
        platform_done_states = []
        steps = []

        space_data_table = self._populate_space_data_table()

        for _, epi_artifact in self.status_bar(self.evaluation_outcome.episode_artifacts.items()):
            episode_metadata.append(
                EpisodeMetaData(
                    test_case=epi_artifact.test_case,
                    frame_rate=epi_artifact.frame_rate,
                    worker_index=epi_artifact.worker_index,
                    wall_time_sec=epi_artifact.duration_sec,
                    steps=len(epi_artifact.steps),
                    seconds=1 / epi_artifact.frame_rate * len(epi_artifact.steps),
                )
            )

            # Process Agent Dones
            agent_done_statuses.extend(self._generate_rows_agent_dones(episode_artifact=epi_artifact))

            # Process Platform Dones
            platform_done_states.extend(self._generate_rows_platform_dones(episode_artifact=epi_artifact))

            # Process Agent Steps
            steps.extend(self._generate_rows_agent_steps(episode_artifact=epi_artifact, space_data_table=space_data_table))

        self.print_func("Populating space data table...")
        self.space_data_table = pd.DataFrame(space_data_table)
        self.print_func("Populating agent done status table...")
        self.agent_dones_table = pd.DataFrame(agent_done_statuses)
        self.print_func("Populating platform done states table...")
        self.platform_dones_table = pd.DataFrame(platform_done_states)
        start = time.time()
        self.print_func("Populating steps table...this could take awhile")
        self.agent_steps_table = pd.DataFrame(steps)
        end = time.time()
        self.print_func(f"|----Took {end-start} seconds")
        self.print_func("Populating episode metadata table...")
        self.episode_metadata_table = pd.DataFrame(episode_metadata)

    def _generate_rows_agent_dones(self, episode_artifact: EpisodeArtifact) -> list[AgentDoneStatus]:  # noqa: PLR6301
        """Generates rows for the agent dones"""
        rows = []
        assert episode_artifact.test_case is not None
        for agent_id, done_status_dict in episode_artifact.episode_state.items():
            for done_name, done_status in done_status_dict.items():
                rows.append(
                    AgentDoneStatus(
                        agent_id=agent_id, done_name=done_name, done_status=done_status.name, test_case=episode_artifact.test_case
                    )
                )
        return rows

    def _generate_rows_platform_dones(self, episode_artifact: EpisodeArtifact) -> list[PlatformDoneStates]:  # noqa: PLR6301
        """Generates rows for the platform dones"""
        row = []
        assert episode_artifact.test_case is not None
        for platform_name, platform_done_dict in episode_artifact.dones.items():
            for done_name, done_state in platform_done_dict.items():
                row.append(
                    PlatformDoneStates(
                        platform_name=platform_name, done_name=done_name, done_triggered=done_state, test_case=episode_artifact.test_case
                    )
                )
        return row

    def _generate_rows_agent_steps(self, episode_artifact: EpisodeArtifact, space_data_table: list[SpaceData]) -> list[FlattenedAgentStep]:
        """Generates rows for the agent steps"""
        row: list[FlattenedAgentStep] = []
        assert episode_artifact.test_case is not None
        if episode_artifact.space_definitions is None:
            warnings.warn("Space definitions missing from episode artifact. Cannot process agent steps.")
            return row

        episode_length = len(episode_artifact.steps)
        for idx, step in enumerate(episode_artifact.steps):
            step_idx = idx + 1
            step_idx_reversed = step_idx - episode_length
            step_time_index_kwargs = {
                "step": step_idx,
                "step_rel_to_end": step_idx_reversed,
                "time": step_idx / episode_artifact.frame_rate if episode_artifact.frame_rate else None,
                "time_rel_to_end": step_idx_reversed / episode_artifact.frame_rate if episode_artifact.frame_rate else None,
            }
            for agent_name, agent_step in step.agents.items():
                platform_names = episode_artifact.agent_to_platforms[agent_name]
                # #############################################################
                # Observations: Creates a row of data for each observation
                # #############################################################
                flattened_obs = self._process_raw_observation(agent_step.observations, multiplatform=len(platform_names) > 1)
                obs_space = episode_artifact.space_definitions.observation_space[agent_name]
                # Process multiplatform observation space. This will get rid of the top-level
                # Repeated space.
                if len(platform_names) > 1:
                    assert isinstance(obs_space, gymnasium.spaces.Dict)
                    obs_space = self._restructure_multiplatform_obs_space(obs_space)

                # Perform a simple flattening (this will not flatten the Repeated spaces in the nested dict)
                # NOTE: we do not call _to_flatten_dict here because we want to preserve the Repeated space
                # types so that we can parse the observation values accordingly below.
                flattened_obs_space = flatten_dict.flatten(obs_space, reducer=self._reducer)
                for obs_name, obs_value in flattened_obs.items():
                    # #########################################################
                    # Handles Repeated Observations
                    # #########################################################
                    if isinstance(flattened_obs_space[obs_name], Repeated):
                        # RepeatedObservations are structured as List[Dict[str, Any]]. Iterate through the repeated
                        # observations. Each repeated observation is processed and flattened into the table.
                        # For each element in the top level list
                        for idx, packed_obs in enumerate(obs_value):  # noqa: PLW2901
                            # Each element should be a dictionary, if it is not, then skip and warn.
                            if not isinstance(packed_obs, dict):
                                warnings.warn(f"Skipping: {obs_name}. Expected dict got: {type(packed_obs)}")
                                continue
                            for key, value in packed_obs.items():
                                # TODO: this key is wrong
                                attribute_name_suffix = f"[{idx}]"
                                value = _return_python_typed_value(_get_single_array_element(value))  # noqa: PLW2901
                                # #############################################
                                # Handles multi-dimensional fields
                                # #############################################
                                row = self._create_row_foreach_array_element(
                                    input_value=value,
                                    stored_rows=row,
                                    attribute_name=f"{obs_name}{self._reducer_map[self._reducer]}{key}",
                                    agent_id=agent_name,
                                    attribute_descrip=FlattenedAgentStep.AttributeDescription.observation,
                                    test_case=episode_artifact.test_case,
                                    space_data_table=space_data_table,
                                    attribute_name_suffix=attribute_name_suffix,
                                    **step_time_index_kwargs,
                                )
                    # #########################################################
                    # Non-Repeated Obs
                    # #########################################################
                    else:
                        obs_value = _return_python_typed_value(_get_single_array_element(obs_value))  # noqa: PLW2901
                        # #####################################################
                        # Handles multi-dimensional fields
                        # #####################################################

                        row = self._create_row_foreach_array_element(
                            input_value=obs_value,
                            attribute_name=obs_name,
                            stored_rows=row,
                            agent_id=agent_name,
                            attribute_descrip=FlattenedAgentStep.AttributeDescription.observation,
                            test_case=episode_artifact.test_case,
                            space_data_table=space_data_table,
                            attribute_name_suffix=None,
                            **step_time_index_kwargs,
                        )
                # #############################################################
                # Actions: Creates a row of data for each action
                # #############################################################
                flattened_actions = self._process_raw_actions(agent_step.actions)
                for action_name, action_value in flattened_actions.items():
                    action_value = _return_python_typed_value(_get_single_array_element(action_value))  # noqa: PLW2901
                    row = self._create_row_foreach_array_element(
                        input_value=action_value,
                        attribute_name=action_name,
                        stored_rows=row,
                        agent_id=agent_name,
                        attribute_descrip=FlattenedAgentStep.AttributeDescription.action,
                        test_case=episode_artifact.test_case,
                        space_data_table=space_data_table,
                        attribute_name_suffix=None,
                        **step_time_index_kwargs,
                    )
                # Rewards: Creates a row for each reward
                if agent_step.rewards:
                    for reward_name, rewards_value in agent_step.rewards.items():
                        row.append(
                            FlattenedAgentStep(
                                agent_id=agent_name,
                                attribute_descrip=FlattenedAgentStep.AttributeDescription.reward,
                                attribute_name=reward_name,
                                attribute_value=rewards_value,
                                test_case=episode_artifact.test_case,
                                step=step_idx,
                                step_rel_to_end=step_idx_reversed,
                                time=step_time_index_kwargs["time"],
                                time_rel_to_end=step_time_index_kwargs["time_rel_to_end"],
                            )
                        )
                # Total Reward: Creates one row for the cumulative reward (total reward)
                row.append(
                    FlattenedAgentStep(
                        agent_id=agent_name,
                        attribute_descrip=FlattenedAgentStep.AttributeDescription.cumulative_reward,
                        attribute_name="total_reward",
                        attribute_value=agent_step.total_reward,
                        test_case=episode_artifact.test_case,
                        step=step_idx,
                        step_rel_to_end=step_idx_reversed,
                        time=step_time_index_kwargs["time"],
                        time_rel_to_end=step_time_index_kwargs["time_rel_to_end"],
                    )
                )
        return row

    def _populate_space_data_table(self):
        """Populates the space data table"""
        # The space definitions and units are the same across all episode
        # artifacts, so we only need to process one.
        rows = []

        epi_artifact = self.evaluation_outcome.episode_artifacts[0]
        space_definitions = epi_artifact.space_definitions

        if space_definitions is not None:
            flattened_obs_units = {}
            for agent_id, _units in epi_artifact.observation_units.items():
                flattened_obs_units[agent_id] = _to_flattened_dict(_units, reducer=self._reducer)

            rows.extend(
                self._generate_rows_space_data_table(
                    space_dict=space_definitions.action_space.spaces,
                    normalized_space_dict=space_definitions.normalized_action_space.spaces,
                    obs_units=flattened_obs_units,
                )
            )

            rows.extend(
                self._generate_rows_space_data_table(
                    space_dict=space_definitions.observation_space.spaces,
                    normalized_space_dict=space_definitions.normalized_observation_space.spaces,
                    obs_units=flattened_obs_units,
                )
            )

        return rows

    def _generate_rows_space_data_table(self, space_dict: OrderedDict, normalized_space_dict: OrderedDict, obs_units: dict):
        """Generates rows for the space data table given the spaces dict and attribute description"""
        rows: list = []
        # Iterate over the agents
        for agent_id, space_def in space_dict.items():
            flattened_space_def = _to_flattened_dict(space_def, reducer=self._reducer)
            flattened_normalized_space_def = _to_flattened_dict(normalized_space_dict[agent_id], reducer=self._reducer)
            obs_unit_one_agent = obs_units.get(agent_id, {})
            # Iterate over the spaces for each agent
            # The space_def comes from the original space. For the observation space, this may contain the sensors that were
            # excluded from training. The normalized_space_dict filters these sensors out.
            for flattened_key, normalized_space in flattened_normalized_space_def.items():
                assert isinstance(normalized_space, gymnasium.Space)
                min_val, max_val = get_min_max_from_gymnasium_space(flattened_space_def[flattened_key])
                min_val_normed, max_val_normed = get_min_max_from_gymnasium_space(normalized_space)
                # Explicit typing for mypy
                if isinstance(min_val, np.ndarray) and isinstance(max_val, np.ndarray):
                    assert isinstance(max_val_normed, np.ndarray) and isinstance(min_val_normed, np.ndarray)
                    rows = self._process_array_attribute_space_data(
                        input_min=min_val,
                        input_max=max_val,
                        input_min_normed=min_val_normed,
                        input_max_normed=max_val_normed,
                        stored_rows=rows,
                        attribute_name=flattened_key,
                        agent_id=agent_id,
                        space_type=normalized_space.__class__.__name__,
                        units=obs_unit_one_agent.get(flattened_key),
                    )
                if isinstance(min_val, int | float) and isinstance(max_val, int | float):
                    assert isinstance(min_val_normed, int | float) and isinstance(max_val_normed, int | float)
                    _units = _get_single_array_element(obs_unit_one_agent.get(flattened_key))
                    rows.append(
                        SpaceData(
                            agent_id=agent_id,
                            attribute_name=flattened_key,
                            space_type=normalized_space.__class__.__name__,
                            min=min_val,
                            max=max_val,
                            normalized_min=min_val_normed,
                            normalized_max=max_val_normed,
                            units=_units,
                        )
                    )
        return rows

    @staticmethod
    def _process_array_attribute_space_data(
        input_min: np.ndarray,
        input_max: np.ndarray,
        input_min_normed: np.ndarray,
        input_max_normed: np.ndarray,
        stored_rows: list[SpaceData],
        attribute_name,
        units: list | (None | str),
        **space_data_kwargs,
    ) -> list[SpaceData]:
        """Helper function to flatten various array fields in the space data table."""
        input_min = input_min.flatten()
        input_max = input_max.flatten()
        input_min_normed = input_min_normed.flatten()
        input_max_normed = input_max_normed.flatten()

        assert len(input_min) == len(input_max)
        n_elements = len(input_min)
        for idx in range(n_elements):
            min_val = _return_python_typed_value(input_min[idx])
            max_val = _return_python_typed_value(input_max[idx])
            min_val_norm = _return_python_typed_value(input_min_normed[idx])
            max_val_norm = _return_python_typed_value(input_max_normed[idx])
            # Creates a zero-padded index for the attribute
            attribute_padded_idx = str(idx).zfill(len(str(n_elements)))
            new_attribute_name = f"{attribute_name}_{attribute_padded_idx }"
            if isinstance(units, list | tuple):
                _unit = units[idx]
            elif isinstance(units, str):
                _unit = units
            elif units is None:
                _unit = None
            else:
                raise ValueError(f"Units is an unknown type: units: {units}, type: {type(units)}")
            stored_rows.append(
                SpaceData(
                    min=min_val,
                    max=max_val,
                    normalized_min=min_val_norm,
                    normalized_max=max_val_norm,
                    attribute_name=new_attribute_name,
                    units=_unit,
                    **space_data_kwargs,
                )
            )
        return stored_rows

    @staticmethod
    def _create_row_foreach_array_element(
        input_value: tuple | (list | (np.ndarray | (float | int))),
        stored_rows: list[FlattenedAgentStep],
        attribute_name: str,
        space_data_table: list[SpaceData] | None = None,
        attribute_name_suffix: str | None = None,
        **flatten_agent_step_kwargs,
    ) -> list[FlattenedAgentStep]:
        """
        Helper function to process the array attributes. Creates a new column
        for each element in the input, where the new attribute name contains an index
        corresponding to the element of its original array. When the input value is not an array,
        a single row is added.
        """
        # #####################################################################
        # Handles multi-dimensional fields
        # #####################################################################
        if isinstance(input_value, np.ndarray | list | tuple):
            if isinstance(input_value, np.ndarray):
                input_value = input_value.flatten()

            n_elements = len(input_value)
            for _attribute_idx, _value in enumerate(input_value):
                _value = _return_python_typed_value(_value)
                # zero pads up to the max number of elements - this makes sorting the attribute names easier
                attribute_idx_str = str(_attribute_idx).zfill(len(str(n_elements)))
                # This is the structure of the attribute name in the space table for arrays
                new_attribute_name = f"{attribute_name}_{attribute_idx_str}"
                if space_data_table is not None:
                    units = None
                    units = [
                        item.units
                        for item in space_data_table
                        if (item.agent_id == flatten_agent_step_kwargs["agent_id"]) and (item.attribute_name == new_attribute_name)
                    ]
                    assert len(units) <= 1
                    unit = units[0] if len(units) > 0 else None
                # When a suffix is passed in (this is for repeated obs), it will go before the array index
                if attribute_name_suffix:
                    new_attribute_name = f"{attribute_name}{attribute_name_suffix}_{attribute_idx_str}"
                stored_rows.append(
                    FlattenedAgentStep(attribute_name=new_attribute_name, attribute_value=_value, units=unit, **flatten_agent_step_kwargs)
                )
            return stored_rows
        # #####################################################################
        # Handle single-valued fields
        # #####################################################################
        if space_data_table is not None:
            units = None
            units = [
                item.units
                for item in space_data_table
                if (item.agent_id == flatten_agent_step_kwargs["agent_id"]) and (item.attribute_name == attribute_name)
            ]
            assert len(units) <= 1
            unit = units[0] if len(units) > 0 else None
        if attribute_name_suffix:
            attribute_name = f"{attribute_name}{attribute_name_suffix}"
        stored_rows.append(
            FlattenedAgentStep(attribute_name=attribute_name, attribute_value=input_value, units=unit, **flatten_agent_step_kwargs)
        )
        return stored_rows

    def _restucture_multiplatform_obs(self, observations: OrderedDict) -> OrderedDict:  # noqa: PLR6301
        """Restructures the multiplatform observaton.
        The multiplatform observations have the structure: OrderedDict(platform_name, [OrderedDict(obs_name, ...)])"""
        observations = copy.deepcopy(observations)
        restructured_obs = OrderedDict()
        error_msg = "The assumption of how the multiplatform observations are structured is incorrect"
        for _platform_name, obs in observations.items():
            assert len(obs) == 1, error_msg
            restructured_obs[_platform_name] = obs[0]
        return restructured_obs

    def _restructure_multiplatform_obs_space(self, obs_space: gymnasium.spaces.Dict) -> gymnasium.spaces.Dict:  # noqa: PLR6301
        """Restructures the multiplatform observation space. The multiplatform observation space is structured as
        {agent name: {platform name: Repeated(Dict(...), ..})}}. This removes
        the top-level Repeated space, so it flattens appropriately"""
        obs_space = copy.deepcopy(obs_space)
        restructured_space = gymnasium.spaces.Dict()
        for _platform_name, _space in obs_space.items():
            assert isinstance(_space, Repeated), f"Expected Repeated Space, got: {type(_space)}"
            restructured_space[_platform_name] = _space.child_space
        return restructured_space

    def _process_raw_observation(self, observation: OrderedDict, multiplatform: bool = False):
        """Processes the raw observation"""
        tmp = observation.copy()
        for idx, val in observation.items():
            for key, obs in val.items():
                if isinstance(obs, Quantity):
                    tmp[idx][key] = obs.m
        observation = tmp

        if multiplatform:
            return flatten_dict.flatten(self._restucture_multiplatform_obs(observation), reducer=self._reducer)
        return flatten_dict.flatten(observation, reducer=self._reducer)

    def _process_raw_actions(self, action: OrderedDict):
        return flatten_dict.flatten(action, reducer=self._reducer)


def _get_single_array_element(input_value: typing.Any):
    """When an iterable contains a single element, returns the element"""
    if isinstance(input_value, np.ndarray):
        if input_value.shape and max(input_value.shape) == 1:
            return input_value[0]
        # In some cases, values are stored as "no length arrays" where the type is an array
        # but it has no shape, when this is the case, calling .tolist() on it returns the underlying
        # value in the array
        if len(input_value.shape) == 0:
            return input_value.tolist()

    if isinstance(input_value, list | tuple) and len(input_value) == 1:
        return input_value[0]

    return input_value


def _return_python_typed_value(input_value: typing.Any):
    """When the input value is a numpy generic type (np.float32, etc.)
    returns the python equivalent"""
    if isinstance(input_value, np.generic):
        return input_value.item()
    return input_value


def get_min_max_from_gymnasium_space(space: gymnasium.Space):
    """Returns the minimum and maximum of a gymnasium space"""
    if isinstance(space, gymnasium.spaces.Discrete):
        min_val = space.start
        max_val = min_val + space.n
        return _return_python_typed_value(min_val), _return_python_typed_value(max_val)

    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        max_val = _get_single_array_element(space.nvec)
        min_val = _get_single_array_element(np.zeros_like(max_val))
        return _return_python_typed_value(min_val), _return_python_typed_value(max_val)

    if isinstance(space, gymnasium.spaces.MultiBinary):
        min_val = _get_single_array_element(np.zeros(shape=space.shape))
        max_val = _get_single_array_element(min_val + 1)
        return _return_python_typed_value(min_val), _return_python_typed_value(max_val)

    if isinstance(space, gymnasium.spaces.Box):
        min_val = _get_single_array_element(space.low)
        max_val = _get_single_array_element(space.high)
        return _return_python_typed_value(min_val), _return_python_typed_value(max_val)

    raise NotImplementedError(f"An unsupported space was provided: {space}")
