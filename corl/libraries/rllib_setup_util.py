"""Auto code for rllib
"""
import typing

from pydantic import BaseModel
from ray.rllib.agents.trainer import COMMON_CONFIG


class AutoRllibConfigSetup(BaseModel):
    """Meta parameters for automatically defining rllib_config settings, these are used
    to override the following keys:
        num_cpus_for_driver

        num_workers
        num_cpus_per_worker

        num_gpus
        num_gpus_per_worker

        rollout_fragment_length
        train_batch_size
        sgd_minibatch_size
    """
    num_trials: int = 1
    gpus_per_worker: bool = False
    sgd_minibatch_size_percentage: float = 0.1
    ignore_hyper_threads: bool = True


def auto_configure_rllib_config(
    rllib_config: typing.Dict[str, typing.Any], auto_rllib_config_setup: AutoRllibConfigSetup, ray_resources: dict
) -> None:
    """Optimize rllib_config parameters for trainer alg"""
    print("*" * 50 + "Auto Updates to RLLIB Settings (if not set)")

    num_trials = auto_rllib_config_setup.num_trials
    sgd_minibatch_size_percentage = auto_rllib_config_setup.sgd_minibatch_size_percentage
    gpus_per_worker = auto_rllib_config_setup.gpus_per_worker

    #
    # note: Only allow gpu to be used if it is avail - This will override
    gpus_available = get_gpu_avail(ray_resources)

    cpus_available = get_cpu_avail(auto_rllib_config_setup, ray_resources)

    #
    # Note: this checks the cpus per trial on system
    if "num_cpus_for_driver" in rllib_config.keys():
        num_cpus_for_driver = rllib_config["num_cpus_for_driver"]
    else:
        num_cpus_for_driver = COMMON_CONFIG["num_cpus_for_driver"]

    cpus_per_trial_available: float = int((cpus_available - num_cpus_for_driver) / num_trials)

    if cpus_per_trial_available == 0:
        raise RuntimeError(
            f"Not enough resourses for the number of trials (int(({cpus_available} - {num_cpus_for_driver}) / {num_trials}))"
        )

    if "num_cpus_per_worker" in rllib_config.keys():
        workers_per_arena = max(1, int(cpus_per_trial_available / rllib_config["num_cpus_per_worker"]))
    else:
        workers_per_arena = max(1, int(cpus_per_trial_available / COMMON_CONFIG["num_cpus_per_worker"]))

    if 'num_workers' not in rllib_config:
        rllib_config['num_workers'] = {"grid_search": [workers_per_arena]}
        print(f"rllib_config['num_workers'] = {rllib_config['num_workers']}")
    else:
        rllib_config['num_workers'] = {"grid_search": [rllib_config['num_workers']]}
        print(f"rllib_config['num_workers'] = {rllib_config['num_workers']} -- no updates")

    if gpus_per_worker:
        num_gpus = 0.0001 * gpus_available
        num_gpus_per_worker = (gpus_available - num_gpus) / workers_per_arena
        rllib_config['num_gpus'] = {"grid_search": [num_gpus]}
        rllib_config['num_gpus_per_worker'] = {"grid_search": [num_gpus_per_worker]}
    else:
        num_gpus = gpus_available / num_trials
        if 'num_gpus' not in rllib_config:
            rllib_config['num_gpus'] = {"grid_search": [num_gpus]}
            print(f"rllib_config['num_gpus'] = {rllib_config['num_gpus']}")
        else:
            rllib_config['num_gpus'] = {"grid_search": [rllib_config['num_gpus']]}
            print(f"rllib_config['num_gpus'] = {rllib_config['num_gpus']} -- no updates")

    update_rollout_fragment_length(rllib_config)

    update_train_batch_size(rllib_config)

    update_sgd_minibatch_size(rllib_config, sgd_minibatch_size_percentage)

    print("*" * 50)


def update_rollout_fragment_length(rllib_config: dict) -> None:
    """Attempts to auto fill the rollout fragment length

    Parameters
    ----------
    rllib_config : dict
        the current config
    """
    if "horizon" in rllib_config.keys():
        temp_horizon = None
        if isinstance(rllib_config['horizon'], dict) and len(rllib_config['horizon']["grid_search"]) == 1:
            temp_horizon = rllib_config['horizon']["grid_search"] = 0
        elif isinstance(rllib_config['horizon'], int):
            temp_horizon = rllib_config['horizon']
        if "rollout_fragment_length" not in rllib_config.keys():
            rllib_config['rollout_fragment_length'] = {"grid_search": [temp_horizon]}
            print(f"rllib_config['rollout_fragment_length'] = {rllib_config['rollout_fragment_length']}")
        else:
            rllib_config['rollout_fragment_length'] = {"grid_search": [rllib_config['rollout_fragment_length']]}
            print(f"rllib_config['rollout_fragment_length'] = {rllib_config['rollout_fragment_length']} -- no updates")


def update_train_batch_size(rllib_config: dict) -> None:
    """Attempts to auto fill the train batch size

    Parameters
    ----------
    rllib_config : dict
        the current config
    """
    if "train_batch_size" not in rllib_config.keys():
        rllib_config['train_batch_size'] = {
            "grid_search": [int(rllib_config['num_workers']['grid_search'][0] * rllib_config['rollout_fragment_length']['grid_search'][0])]
        }
        print(f"rllib_config['train_batch_size'] = {rllib_config['train_batch_size']}")
    else:
        rllib_config['train_batch_size'] = {"grid_search": [rllib_config['train_batch_size']]}
        print(f"rllib_config['train_batch_size'] = {rllib_config['train_batch_size']} -- no updates")


def update_sgd_minibatch_size(rllib_config: dict, sgd_minibatch_size_percentage: float) -> None:
    """Attempts to auto fill the sgd_minibatch_size

    Parameters
    ----------
    rllib_config : dict
        the current config
    """
    if "sgd_minibatch_size" not in rllib_config.keys():
        rllib_config['sgd_minibatch_size'] = {
            "grid_search": [int(rllib_config['train_batch_size']['grid_search'][0] * sgd_minibatch_size_percentage)]
        }
        print(f"rllib_config['sgd_minibatch_size'] = {rllib_config['sgd_minibatch_size']} ")
    else:
        rllib_config['sgd_minibatch_size'] = {"grid_search": [rllib_config['sgd_minibatch_size']]}
        print(f"rllib_config['sgd_minibatch_size'] = {rllib_config['sgd_minibatch_size']} -- no updates")


def get_gpu_avail(ray_resources: dict) -> float:
    """get the gpu avail

    Parameters
    ----------
    ray_resources : dict
        resources dict

    Returns
    -------
    float
        num gpu
    """
    if "GPU" in ray_resources.keys():
        gpus_available = ray_resources["GPU"]
    else:
        gpus_available = COMMON_CONFIG["num_gpus"]
    return gpus_available


def get_cpu_avail(auto_rllib_config_setup, ray_resources: dict) -> float:
    """get the cpu avail

    Parameters
    ----------
    ray_resources : dict
        resources dict

    Returns
    -------
    float
        num cpu
    """
    cpus_available: float = 1
    if auto_rllib_config_setup.ignore_hyper_threads:
        cpus_available = ray_resources["CPU"] / 2
    else:
        cpus_available = ray_resources["CPU"]

    return cpus_available
