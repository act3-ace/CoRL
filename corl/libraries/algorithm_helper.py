from ray.rllib.algorithms.algorithm import Algorithm

from corl.environment.multi_agent_env import ACT3MultiAgentEnv


def reset_epp(algorithm: Algorithm):
    """Resets all episode parameter providers"""
    worker_set = algorithm.workers
    if worker_set is None:
        raise RuntimeError("No workers available")

    def reset_env_epp(env) -> list:
        if isinstance(env, ACT3MultiAgentEnv):
            for epp in env.config.epp_registry.values():
                epp.reset()
        return []

    worker_set.foreach_env(reset_env_epp)


def cleanup_algorithm(algorithm: Algorithm):
    """Explicitly deletes all environments in algorithm to ensure that they can be garbage collected"""
    algorithm.stop()

    def try_shutdown(env):
        if isinstance(env, ACT3MultiAgentEnv):
            env.simulator.shutdown()

    if algorithm.workers is not None:
        try_shutdown(algorithm.workers.local_worker().env)
        del algorithm.workers.local_worker().env

        try_shutdown(algorithm.workers.local_worker().config.env)
        del algorithm.workers.local_worker().config.env

        try:
            for env in algorithm.workers.local_worker().async_env.envs:
                try_shutdown(env)

            del algorithm.workers.local_worker().async_env.envs[:]
        except AttributeError:
            pass

        try:
            for env_state in algorithm.workers.local_worker().async_env.env_states:
                try_shutdown(env_state.env)
                del env_state.env
        except AttributeError:
            pass

        try:
            try_shutdown(algorithm.workers.local_worker().async_env._unwrapped_env)  # noqa: SLF001
            del algorithm.workers.local_worker().async_env._unwrapped_env  # noqa: SLF001
        except AttributeError:
            pass
