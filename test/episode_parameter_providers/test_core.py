import pickle
import typing

import flatten_dict
import numpy as np
import pytest
from pydantic import PositiveInt

from corl.episode_parameter_providers import (
    EpisodeParameterProvider,
    EpisodeParameterProviderValidator,
    ParameterModel,
    PathLike,
    Randomness,
)
from corl.episode_parameter_providers.remote import RemoteEpisodeParameterProvider
from corl.episode_parameter_providers.simple import SimpleParameterProvider
from corl.libraries.cleanup import cleanup
from corl.libraries.parameters import ConstantParameter, PassthroughParameterWrapper
from corl.libraries.units import corl_get_ureg
from corl.test_utils.testing_ureg_func import ureg_setup_func


@pytest.mark.parametrize("is_remote", [False, True])
def test_wrapper_simple(is_remote, tmp_path):
    rng = np.random.default_rng(seed=0)

    seed_params = {
        "param1param2": PassthroughParameterWrapper(
            wrapped={
                "param1": ConstantParameter(name="param1", units="dimensionless", value=3),
                "param2": ConstantParameter(name="param2", units="dimensionless", value=5),
            }
        )
    }

    flat_params = flatten_dict.flatten(seed_params)

    if is_remote:
        epp = RemoteEpisodeParameterProvider(internal_class=SimpleParameterProvider, parameters=flat_params, actor_name="gheewiz")
        _epp_cleanup_handle = cleanup(lambda: epp.kill_actor())
    else:
        epp = SimpleParameterProvider(parameters=flat_params)

    # value = 0
    for _ in range(5):
        for _ in range(10):
            params, _, _ = epp.get_params(rng)
            assert len(params) == 1
            assert ("param1param2",) in params
            saved_dict = params[("param1param2",)].get_value(rng, {})
            assert saved_dict == {
                "param1": corl_get_ureg().Quantity(value=3, units="dimensionless"),
                "param2": corl_get_ureg().Quantity(value=5, units="dimensionless"),
            }
            assert "param2" in saved_dict
            assert "param1" in saved_dict

        assert epp.compute_metrics() == {}
        epp.update({}, rng)

    # These do not do anything, so it just checks that they do not fail
    filename = tmp_path / "checkpoint.pkl"
    epp.save_checkpoint(filename)
    epp.load_checkpoint(filename)


@pytest.mark.parametrize("is_remote", [False, True])
def test_simple(is_remote, tmp_path):
    rng = np.random.default_rng(seed=0)

    seed_params = {
        "param1": ConstantParameter(name="param1", units="dimensionless", value=3),
        "group": {"param2": ConstantParameter(name="param2", units="dimensionless", value=5)},
    }
    flat_params = flatten_dict.flatten(seed_params)

    if is_remote:
        epp = RemoteEpisodeParameterProvider(internal_class=SimpleParameterProvider, parameters=flat_params, actor_name="gheewiz")
        _epp_cleanup_handle = cleanup(lambda: epp.kill_actor())
    else:
        epp = SimpleParameterProvider(parameters=flat_params)

    # value = 0
    for _ in range(5):
        for _ in range(10):
            params, _, _ = epp.get_params(rng)
            assert len(params) == 2
            assert ("param1",) in params
            assert params[("param1",)].get_value(rng, {}) == corl_get_ureg().Quantity(value=3, units="dimensionless")
            assert ("group", "param2") in params
            assert params[("group", "param2")].get_value(rng, {}) == corl_get_ureg().Quantity(value=5, units="dimensionless")

        assert epp.compute_metrics() == {}
        epp.update({}, rng)

    # These do not do anything, so it just checks that they do not fail
    filename = tmp_path / "checkpoint.pkl"
    epp.save_checkpoint(filename)
    epp.load_checkpoint(filename)


class IncrementingConstantValidator(EpisodeParameterProviderValidator):
    update_increment: PositiveInt


class IncrementingConstant(EpisodeParameterProvider):
    def __init__(self, **kwargs) -> None:
        ureg_setup_func()

        self.config: IncrementingConstantValidator
        super().__init__(**kwargs)

        assert all(isinstance(x, ConstantParameter) for x in self.config.parameters.values())
        self._value = 0
        self._episode_id = 0

    def reset(self):
        self._value = 0
        self._episode_id = 0

    @staticmethod
    def get_validator() -> type[IncrementingConstantValidator]:
        return IncrementingConstantValidator

    def _do_get_params(self, rng: Randomness, env_epp_ctx: dict | None = None) -> tuple[ParameterModel, int | None, dict | None]:
        episode_id_out = self._episode_id
        self._episode_id += 1
        return (
            {
                name: ConstantParameter(
                    name=name[-1], units=self.config.parameters[name].config.units, value=seed_param.get_value(rng, {}).m + self._value
                )
                for name, seed_param in self.config.parameters.items()
            },
            episode_id_out,
            env_epp_ctx,
        )

    def compute_metrics(self) -> dict[str, typing.Any]:
        return {"value": self._value}

    def update(self, results: dict, rng: Randomness) -> None:
        self._value += self.config.update_increment

    def save_checkpoint(self, checkpoint_path: PathLike) -> None:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(self._value, f)

    def load_checkpoint(self, checkpoint_path: PathLike) -> None:
        with open(checkpoint_path, "rb") as f:
            self._value = pickle.load(f)  # noqa: S301


@pytest.mark.parametrize("is_remote", [False, True])
def test_extensive(is_remote, tmp_path):
    rng = np.random.default_rng(seed=0)

    seed_params = {
        "param1": ConstantParameter(name="param1", units="dimensionless", value=3),
        "group": {"param2": ConstantParameter(name="param2", units="dimensionless", value=5)},
    }
    flat_params = flatten_dict.flatten(seed_params)

    if is_remote:
        epp = RemoteEpisodeParameterProvider(
            internal_class=IncrementingConstant, internal_config={"update_increment": 17}, parameters=flat_params, actor_name="gheewiz"
        )
        _epp_cleanup_handle = cleanup(lambda: epp.kill_actor())
    else:
        epp = IncrementingConstant(parameters=flat_params, update_increment=17)

    value = 0
    index = 0
    for _ in range(5):
        for _ in range(10):
            params, episode_id, _ = epp.get_params(rng)
            assert len(params) == 2
            assert ("param1",) in params
            assert params[("param1",)].get_value(rng, {}) == corl_get_ureg().Quantity(value=3 + value, units="dimensionless")
            assert ("group", "param2") in params
            assert params[("group", "param2")].get_value(rng, {}) == corl_get_ureg().Quantity(value=5 + value, units="dimensionless")
            assert episode_id == index
            index += 1

        assert epp.compute_metrics() == {"value": value}
        epp.update({}, rng)
        value += 17

    filename = tmp_path / "checkpoint.pkl"

    epp.save_checkpoint(filename)
    rng1 = np.random.default_rng(seed=1)
    params1, _, _ = epp.get_params(rng1)

    rng2 = np.random.default_rng(seed=1)
    if is_remote:
        epp2 = RemoteEpisodeParameterProvider(
            internal_class=IncrementingConstant, internal_config={"update_increment": 17}, parameters=flat_params, actor_name="gheewiz2"
        )

        _epp2_cleanup_handle = cleanup(lambda: epp2.kill_actor())
    else:
        epp2 = IncrementingConstant(parameters=flat_params, update_increment=17)

    epp2.load_checkpoint(filename)
    params2, _, _ = epp2.get_params(rng2)

    assert params1.keys() == params2.keys()
    for k in params1:
        assert params1[k].get_value(rng1, {}) == params2[k].get_value(rng2, {})

    epp.update({}, rng1)
    epp2.update({}, rng2)

    params1, _, _ = epp.get_params(rng1)
    params2, _, _ = epp2.get_params(rng2)

    assert params1.keys() == params2.keys()
    for k in params1:
        assert params1[k].get_value(rng1, {}) == params2[k].get_value(rng2, {})
