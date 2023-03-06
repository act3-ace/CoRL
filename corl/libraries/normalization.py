"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Normalizer
"""
import abc
import copy
import typing
from collections import OrderedDict

import gym.spaces
import numpy as np
from pydantic import BaseModel, validator

from corl.libraries.env_space_util import EnvSpaceUtil


class Normalizer(abc.ABC):
    """Class defining normalization of space and samples
    """

    def __init__(self, **kwargs):
        self.config = self.get_validator(**kwargs)

    @property
    def get_validator(self):
        """
        get validator for this Normalizer
        """

    @abc.abstractmethod
    def normalize_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """
        normalize_space

        Parameters:
            space: gym.spaces.Space
                unnormalized space to be normalized
        Returns:
            gym.spaces.Space
                Normalized space
        """

    @abc.abstractmethod
    def normalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        normalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                unnormalized sample space to be normalized
        Returns:
            EnvSpaceUtil.sample_type
                Normalized sample
        """

    @abc.abstractmethod
    def unnormalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        unnormalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                normalized sample space to be unnormalized
        Returns:
            EnvSpaceUtil.sample_type
                Unnormalized sample
        """


class LinearNormalizerValidator(BaseModel):
    """LinearNormalizerValidator

    Parameters:
        minimum: float
            minimum normalized value
        maximum: float
            maximum normalized value
    """
    minimum: float = -1.0
    maximum: float = 1.0


class LinearNormalizer(Normalizer):
    """Class defining linear normalization of space and samples
    """

    def __init__(self, **kwargs) -> None:
        self.config: LinearNormalizerValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[LinearNormalizerValidator]:
        """get validator for this LinearNormalizer

        Returns:
            LinearNormalizerValidator
        """
        return LinearNormalizerValidator

    def normalize_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """
        normalize_space

        Parameters:
            space: gym.spaces.Space
                unnormalized space to be normalized
        Returns:
            gym.spaces.Space
                Normalized space between minimum and maximum
        """

        return EnvSpaceUtil.normalize_space(space=space, out_min=self.config.minimum, out_max=self.config.maximum)

    def normalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        normalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                unnormalized sample space to be normalized
        Returns:
            EnvSpaceUtil.sample_type
                Normalized sample between min and max output value
        """
        return EnvSpaceUtil.scale_sample_from_space(
            space=space, space_sample=sample, out_min=self.config.minimum, out_max=self.config.maximum
        )

    def unnormalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        unnormalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                normalized sample space from min to max vale be unnormalized
        Returns:
            EnvSpaceUtil.sample_type
                Unnormalized sample
        """
        return EnvSpaceUtil.unscale_sample_from_space(
            space=space, space_sample=sample, out_min=self.config.minimum, out_max=self.config.maximum
        )


class StandardNormalNormalizerValidator(BaseModel):
    """StandardNormalNormalizerValidator

    Parameters:
        mu: float
            mean of distribution
        sigma: float
            standard devivation
    """
    mu: typing.List[float] = [0.0]
    sigma: typing.List[float] = [1.0]

    @validator('mu', 'sigma', pre=True)
    def check_iterable(cls, v):
        """
        Check if mu, sigma are iterable.
        """
        if isinstance(v, float):
            v = [v]
        return v


class StandardNormalNormalizer(Normalizer):
    """Class defining standard norm normalization of space and samples
    """

    def __init__(self, **kwargs) -> None:
        self.config: StandardNormalNormalizerValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[StandardNormalNormalizerValidator]:
        """get validator for this StandardNormalNormalizer

        Returns:
            StandardNormalNormalizerValidator
        """
        return StandardNormalNormalizerValidator

    def normalize_space(self, space: gym.spaces.Space) -> gym.spaces.Space:
        """
        normalize_space

        Parameters:
            space: gym.spaces.Space
                unnormalized space to be normalized
        Returns:
            gym.spaces.Space
                Normalized space between minimum and maximum
        """
        return EnvSpaceUtil.iterate_over_space(
            func=StandardNormalNormalizer.__standard_normalize_space,
            space=space,
            mu=self.config.mu,
            sigma=self.config.sigma,
        )

    def normalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """normalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                unnormalized sample space to be normalized

        Returns:
            EnvSpaceUtil.sample_type
                Normalized sample between min and max output value
        """
        return EnvSpaceUtil.iterate_over_space_with_sample(
            func=StandardNormalNormalizer.standard_normalize_sample, space=space, sample=sample, mu=self.config.mu, sigma=self.config.sigma
        )

    def unnormalize_sample(self, space: gym.spaces.Space, sample: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        """
        unnormalize_sample

        Parameters:
            space: gym.spaces.Space
                unnormalized space
            sample: EnvSpaceUtil.sample_type
                normalized sample space from min to max vale be unnormalized
        Returns:
            EnvSpaceUtil.sample_type
                Unnormalized sample
        """
        tmp = EnvSpaceUtil.iterate_over_space_likes(
            func=StandardNormalNormalizer.unnormalize_sample_from_mu_sigma,
            space_likes=(space, sample),
            mu=self.config.mu,
            sigma=self.config.sigma,
            return_space=False,
        )
        if not isinstance(tmp, gym.spaces.Space):
            return tmp
        raise RuntimeError("bug spat, should never happen")

    @staticmethod
    def __standard_normalize_space(
        space_likes: typing.Tuple[gym.spaces.Space, EnvSpaceUtil.sample_type],
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> gym.spaces.Space:
        """
        Normalizes a given gym box using the provided mu and sigma.

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space]
            The gym space to turn all boxes into the scaled space.
        mu: float = 0.0
            Mu for normalization.
        sigma: float = 1.0
            Sigma for normalization.

        Returns
        -------
        gym.spaces.Space:
            The new gym spaces where all boxes have had their bounds changed.
        """
        space_arg = space_likes[0]
        if isinstance(space_arg, gym.spaces.Box):
            low = np.divide(np.subtract(space_arg.low, mu), sigma)
            high = np.divide(np.subtract(space_arg.high, mu), sigma)
            return gym.spaces.Box(low=low, high=high, shape=space_arg.shape, dtype=np.float32)
        return copy.deepcopy(space_arg)

    @staticmethod
    def standard_normalize_sample(
        space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
        mu: float = 0.0,
        sigma: float = 1,
    ) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        """
        This normalizes a sample from a box space using the mu and sigma arguments.

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space, sample_type]
            The first element is the gym space.
            The second element is the sample of this space to scale.
        mu: float
            The mu used for normalizing the sample.
        sigma: float
            The sigma used for normalizing the sample.

        Returns
        -------
        typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
            The normalized sample.
        """
        (space_arg, space_sample_arg) = space_likes
        if isinstance(space_arg, gym.spaces.Box):
            val = np.array(space_sample_arg)
            norm_value = np.subtract(val, mu)
            norm_value = np.divide(norm_value, sigma)
            return norm_value.astype(np.float32)
        return copy.deepcopy(space_sample_arg)

    @staticmethod
    def unnormalize_sample_from_mu_sigma(
        space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
        mu: float = 0.0,
        sigma: float = 1,
    ) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        """
        This unnormalizes a sample from a box space using the mu and sigma arguments.

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space, sample_type]
            The first element is the gym space.
            The second element is the sample of this space to scale.
        mu: float
            The mu used for unnormalizing the sample.
        sigma: float
            The sigma used for unnormalizing the sample.

        Returns
        -------
        typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
            The unnormalized sample.
        """
        (space_arg, space_sample_arg) = space_likes
        if isinstance(space_arg, gym.spaces.Box):
            val = np.array(space_sample_arg)
            norm_value = np.add(np.multiply(val, sigma), mu)
            return norm_value.astype(np.float32)
        return copy.deepcopy(space_sample_arg)
