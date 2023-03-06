"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Pickle Pong State Callback
"""

import copy
import os
import pickle

from ray.rllib import BaseEnv
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode


class PongPickleEnvState(DefaultCallbacks):
    """
    This callback saves the environment state every fixed number of episodes
    (`save_every_n_episodes`). The accumulated awards at each time step and
    all contents of `env.state` are stored.

    The outputs are saved in the environment config's output path.

    """
    save_every_n_episodes: int = 100
    episode_counter: int = 0
    game_status_key: str = 'game_status'
    rewards_accumulator_key: str = 'rewards_accumulator'
    env_state_key: str = 'env_state'

    def on_episode_step(
        self,
        *,
        base_env: BaseEnv,
        episode: Episode,
        **_,
    ) -> None:
        """Saves the environment state and accumulated rewards into
        `episode.user_data[self.env_state_key]`
        """

        if self.episode_counter % self.save_every_n_episodes == 0:
            env = base_env.get_sub_environments()[episode.env_id]
            if self.env_state_key not in episode.user_data:
                episode.user_data[self.env_state_key] = []

            env_state = copy.deepcopy(env.state)

            # saves the entire environment state object as a dictionary
            # and the accumulated reward at the this step.
            env_state = dict(env_state)
            env_state.update({self.rewards_accumulator_key: dict(episode.user_data[self.rewards_accumulator_key])})
            episode.user_data[self.env_state_key].append(env_state)

    def on_episode_end(self, *, worker, episode: Episode, **_) -> None:
        """
        Pickles the data and writes it out
        """
        if self.env_state_key in episode.user_data:

            env_states = episode.user_data[self.env_state_key]
            game_status = env_states[-1].get(self.game_status_key, '')
            if hasattr(game_status, 'name'):
                game_status = game_status.name

            # Loop through the last state and grab the ball hit counts
            ball_hits_total = 0
            for _, simulator in env_states[-1]['sim_platforms'].items():
                ball_hits_total += simulator.paddle.ball_hits

            count_as_string = str(self.episode_counter).zfill(8)
            output_directory = worker.env.config.output_path / self.env_state_key
            os.makedirs(output_directory, exist_ok=True)
            output_file = output_directory / f'episode_{count_as_string}_len_{len(env_states)}_ballhits_{ball_hits_total}_{game_status}.pkl'
            with open(output_file, "wb") as f:
                pickle.dump(env_states, f)
            print(f'Saved episode to: {output_file}')

            del episode.user_data[self.env_state_key]

        self.episode_counter += 1
