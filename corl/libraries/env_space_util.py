# pylint: disable=too-many-lines
"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
ENV Space Util Module

this file is a mypy nightmare due to lots of typing unions and complex types
do not trust mypy, trust the unit tests
"""
import copy
import logging
import typing
from collections import OrderedDict
from collections.abc import MutableSequence, Sequence

import flatten_dict
import gym
import numpy as np
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.spaces.repeated import Repeated

from corl.libraries.state_dict import StateDict


class EnvSpaceUtil:  # pylint: disable=R0904
    """ ENV Space Util
    """
    _logger = logging.getLogger("EnvSpaceUtil")
    sample_type = typing.Union[OrderedDict, dict, tuple, np.ndarray, list]

    @staticmethod
    def deep_sanity_check_space_sample(  # pylint: disable=R0912
        space: gym.spaces.Space,
        sample: sample_type,
        key_stack: str = "",
    ) -> None:
        """Ensure space sample is consistent with space. This will give a traceback of the exact space that failed

        Parameters
        ----------
        space: gym.spaces.Dict
            the space we expect the sample to conform to
        sample: sample_type
            sample that we are checking if it belongs to the given space
        key_stack: str
            string of the keys we have used when getting to this current spot in the observation_space, observation
            this is used for recursive calls do not set in the initial call to this function
        """
        if isinstance(space, gym.spaces.Dict):
            if not isinstance(sample, (StateDict, OrderedDict, dict)):
                raise ValueError(
                    f"space{key_stack}={space} was a gym.spaces.Dict type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a StateDict, OrderedDict, or dict\n"
                )
            for key, value in sample.items():
                EnvSpaceUtil.deep_sanity_check_space_sample(space.spaces[key], value, key_stack=f"{key_stack}[{key}]")
        elif isinstance(space, gym.spaces.Tuple):
            if not isinstance(sample, tuple):
                raise ValueError(
                    f"space{key_stack}={space} is a gym.spaces.Tuple type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a tuple type"
                )
            for idx, value in enumerate(sample):
                EnvSpaceUtil.deep_sanity_check_space_sample(space.spaces[idx], value, key_stack=f"{key_stack}[{idx}]")
        elif isinstance(space, gym.spaces.Discrete):
            if not isinstance(sample, (int, np.integer, np.ndarray)):
                raise ValueError(
                    f"space{key_stack}={space} is a gym.spaces.Discrete type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not an int or np.integer or np.ndarray"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however "
                    f"space{key_stack} has space definition of {space} {space.n}"
                )
        elif isinstance(space, gym.spaces.Box):
            if not isinstance(sample, (np.ndarray, list, np.floating)):
                raise ValueError(
                    f"space{key_stack}={space} is a gym.spaces.Box type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.float type"
                )
            if not space.contains(sample):
                sample_dtype = getattr(sample, 'dtype', None)
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however "
                    f"space{key_stack} has space definition of {space} {space.low} {space.high} "
                    f"space dtype is {space.dtype}, sample dtype is {sample_dtype}"
                )
        elif isinstance(space, gym.spaces.MultiBinary):
            if not isinstance(sample, (np.ndarray, list, np.integer)):
                raise ValueError(
                    f"space{key_stack}={space} is a gym.spaces.MultiBinary type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.integer type"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however "
                    f"space{key_stack} has space definition of {space} {space.n}"
                )
        elif isinstance(space, gym.spaces.MultiDiscrete):
            if not isinstance(sample, (np.ndarray, list, np.integer)):
                raise ValueError(
                    f"space{key_stack}={space} is a gym.spaces.MultiDiscrete type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.integer type"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however "
                    f"space{key_stack} has space definition of {space} {space.nvec}"
                )
        elif isinstance(space, Repeated):
            if not isinstance(sample, list):
                raise ValueError(
                    f"space{key_stack}={space} is a ray.rllib.utils.spaces.repeated.Repeated type but\n"
                    f"sample{key_stack}={sample}\n"
                    f"is a {type(sample)} type and not a list  type"
                )
            for idx, item in enumerate(sample):
                EnvSpaceUtil.deep_sanity_check_space_sample(space.child_space, item, key_stack=f"{key_stack}[{idx}]")

    @staticmethod
    def sanity_check_space_sample(space, sample):
        """[summary]

        Parameters
        ----------
        space : [type]
            [description]
        sample : [type]
            [description]

        Raises
        ------
        RuntimeError
            [description]
        """
        if not space.contains(sample):
            raise RuntimeError(f"sample of {sample} does not meet space {space} setup")

    @staticmethod
    def deep_merge_dict(source: dict, destination: dict):
        """
        Merget two dictionaries that also can contain sub dictionaries. This function returns the second dict
        but it also modifies it in place

        run me with nosetests --with-doctest file.py

        >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
        >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
        >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
        True
        """
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, OrderedDict())
                EnvSpaceUtil.deep_merge_dict(value, node)
            else:
                destination[key] = value

        return destination

    @staticmethod
    def scale_space(space: gym.spaces.Space, scale: float) -> gym.spaces.Space:
        """
        Multiplies the low and high properties of all the Boxes in the given gym space by the scale input
        Parameters
        ----------
        space: gym.spaces.Space
            the gym space to scale the Boxes of
        scale: float
            what to multiply the Box low and high by
        Returns
        -------
        gym.spaces.Space
            the scaled gym space
        """
        # TODO: this copy probably doesn't actually work but I can dream
        val = copy.deepcopy(space)
        if isinstance(space, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.scale_space(value, scale=scale)
            val = gym.spaces.Dict(spaces=new_dict)
        elif isinstance(space, gym.spaces.Tuple):
            new_thing = [EnvSpaceUtil.scale_space(sp, scale=scale) for sp in space.spaces]
            val = gym.spaces.Tuple(tuple(new_thing))
        elif isinstance(space, gym.spaces.Box):
            scaled_box = gym.spaces.Box(
                low=np.multiply(space.low, scale).astype(np.float32),
                high=np.multiply(space.high, scale).astype(np.float32),
                shape=space.shape,
                dtype=np.float32
            )
            val = scaled_box

        return val

    @staticmethod
    def zero_mean_space(space: gym.spaces.Space) -> gym.spaces.Space:
        """
        Returns a space object where every Box instance has its low and high shifted to be zero mean
        Parameters
        ----------
        space: gym.spaces.Space
            The gym space to zero mean

        Returns
        -------
        gym.spaces.Space
            A gym space the same as the input but with the Box instances shifted
        """
        # TODO: this copy doesn't actually work but I can dream
        val = copy.deepcopy(space)
        if isinstance(space, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.zero_mean_space(value)
            val = gym.spaces.Dict(spaces=new_dict)
        elif isinstance(space, gym.spaces.Tuple):
            new_thing = [EnvSpaceUtil.zero_mean_space(sp) for sp in space.spaces]
            val = gym.spaces.Tuple(tuple(new_thing))
        elif isinstance(space, gym.spaces.Box):
            mean = (space.high + space.low) / 2
            zero_mean_box = gym.spaces.Box(low=space.low - mean, high=space.high - mean, shape=space.shape, dtype=np.float32)
            val = zero_mean_box

        return val

    @staticmethod
    def space_box_min_maxer(
        space_likes: typing.Tuple[gym.spaces.Space],
        out_min: float = -1.0,
        out_max: float = 1.0,
    ) -> gym.spaces.Space:
        """
        Makes a gym box to the out_min and out_max range

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space]
            the gym space to turn all boxes into the scaled space
        out_min: float
            the new low for the boxes
        out_max: float
            the new high for the boxes

        Returns
        -------
        gym.spaces.Space:
            the new gym spaces where all boxes have had their bounds changed
        """
        space_arg = space_likes[0]
        if isinstance(space_arg, gym.spaces.Box):
            return gym.spaces.Box(low=out_min, high=out_max, shape=space_arg.shape, dtype=np.float32)
        return copy.deepcopy(space_arg)

    @staticmethod
    def normalize_space(
        space: gym.spaces.Space,
        out_min=-1,
        out_max=1,
    ) -> gym.spaces.Space:
        """
        This is a convenience wrapper for box_scaler

        Parameters
        ----------
        space: gym.spaces.Space
            the gym space to turn all boxes into the scaled space
        out_min: float
            the new low for the boxes
        out_max: float
            the new high for the boxes

        Returns
        -------
        gym.spaces.Space:
            the new gym spaces where all boxes have had their bounds changed
        """
        return EnvSpaceUtil.iterate_over_space(
            func=EnvSpaceUtil.space_box_min_maxer,
            space=space,
            out_min=out_min,
            out_max=out_max,
        )

    @staticmethod
    def get_zero_sample_from_space(space: gym.spaces.Space) -> sample_type:
        """
        Given a gym space returns an instance of that space but instead of sampling from the
        gym space, returns all zeros. If the space is not a Box and we cannot iterate over it
        then we will sample from it.

        Parameters
        ----------
        space: gym.spaces.Space
            The gym space to zero sample from

        Returns
        -------
        sample_type
            The instance of the gym space but all Box spaces are sampled as zero
        """
        val = space.sample()
        if isinstance(space, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.get_zero_sample_from_space(value)
            val = new_dict
        elif isinstance(space, gym.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.get_zero_sample_from_space(sp) for sp in space.spaces]
            val = tuple(new_tuple)
        elif isinstance(space, gym.spaces.Box):
            val = np.zeros(shape=space.shape, dtype=np.float32)
        elif isinstance(space, gym.spaces.Discrete):
            val = 0
        return val

    @staticmethod
    def get_mean_sample_from_space(space: gym.spaces.Space) -> sample_type:
        """
        Given a gym space returns an instance of that space but instead of sampling from the
        gym space, returns all zeros. If the space is not a Box and we cannot iterate over it
        then we will sample from it.

        Parameters
        ----------
        space: gym.spaces.Space
            The gym space to zero sample from

        Returns
        -------
        sample_type
            The instance of the gym space but all Box spaces are sampled as zero
        """
        val = space.sample()
        if isinstance(space, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.get_mean_sample_from_space(value)
            val = new_dict
        elif isinstance(space, gym.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.get_mean_sample_from_space(sp) for sp in space.spaces]
            val = tuple(new_tuple)
        elif isinstance(space, gym.spaces.Box):
            val = (space.high + space.low) / 2.0
        return val

    @staticmethod
    def add_space_samples(
        space_template: gym.spaces.Space,
        space_sample1: sample_type,
        space_sample2: sample_type,
    ) -> sample_type:
        """
        Adds together two instances of gym spaces. This only adds the ndarray or list types (that were sampled from Box)
        If the object is not a ndarray or list then the value of space_sample1 is returned by default
        Parameters
        ----------
        space_template: gym.spaces.Space
            The template to use for adding these space instances.
            This is to determine the difference between a Box and Discrete or MultiDiscrete or MultiBinary
        space_sample1: sample_type
            The first instance to add
        space_sample2: sample_type
            The second instance to add
        Returns
        -------
        sample_type
            an instance of the space object but with all the Box types added

        """
        # if not type(space_sample1) == type(space_sample2):
        #     raise ValueError('space instances must be of same type')
        # TODO: I want to check they are the same type but dict and OrderedDict should match which makes this annoying
        val: EnvSpaceUtil.sample_type
        if isinstance(space_template, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, space_value in space_template.spaces.items():
                value1 = space_sample1[key]
                value2 = space_sample2[key]
                new_dict[key] = EnvSpaceUtil.add_space_samples(space_value, value1, value2)
            val = new_dict
        elif isinstance(space_template, gym.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.add_space_samples(*args) for args in zip(space_template, space_sample1, space_sample2)]
            val = tuple(new_tuple)
        elif isinstance(space_template, gym.spaces.Box):
            if isinstance(space_sample1, np.ndarray):
                val = np.array(space_sample1 + space_sample2)
            elif isinstance(space_sample1, list):
                val = [value1 + value2 for value1, value2 in zip(space_sample1, space_sample2)]
        else:
            val = copy.deepcopy(space_sample1)
        return val

    @staticmethod
    def clip_space_sample_to_space(space_sample: sample_type, space: gym.spaces.Space, is_wrap: bool = False) -> sample_type:
        """
        Clips a space instance to a given space. After this the space should contain the space instance

        Parameters
        ----------
        space_sample: sample_type
            the space instance we are going to clip
        space: gym.spaces.Space
            the gym space to clip the instance to.
        Returns
        -------
        sample_type
            the clipped space instance
        """
        val = space_sample
        if isinstance(space, Repeated):
            new_obs = []
            for item in val:
                new_obs.append(EnvSpaceUtil.clip_space_sample_to_space(item, space.child_space, is_wrap=is_wrap))
            val = new_obs
        if isinstance(space, gym.spaces.Dict):
            new_dict = OrderedDict()
            for key, space_value in space.spaces.items():
                space_sample_value = space_sample[key]
                new_dict[key] = EnvSpaceUtil.clip_space_sample_to_space(space_sample_value, space_value, is_wrap)
            val = new_dict
        elif isinstance(space, gym.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.clip_space_sample_to_space(siv, sv, is_wrap) for siv, sv in zip(space_sample, space)]
            val = tuple(new_tuple)
        elif isinstance(space, gym.spaces.Box):
            if is_wrap:
                # Takes care of the case where the controls wrap at the min/max value
                # Example: Range = 0-1, value = 1.1 ----> clipping puts to .1
                if space_sample > space.high:
                    val = space.low + (space_sample - space.high)
                elif space_sample < space.low:
                    val = space.high - (space.low - space_sample)
            else:
                # Takes care of the case where the controls saturate at the min/max value
                # Example: Range = 0-1, value = 1.1 ----> clipping puts to 1
                assert isinstance(space_sample, (Sequence, np.ndarray)), \
                    f'Box spaces must have sequence samples, received {type(space_sample).__name__}'
                val = np.clip(a=space_sample, a_min=space.low, a_max=space.high)
        return val

    @staticmethod
    def iterate_over_space(func, space: gym.spaces.Space, *func_args, **func_kwargs) -> gym.spaces.Space:
        """
        utility tool for iterate over space to help mypy handle the return type correctly
        and be a little easier to use

        Args:
            func (_type_): function to call on space and all sub space, will recieve
                            func_args and func_kwargs as arguments
            space (gym.spaces.Space): space to call func on

        Raises:
            RuntimeError: if this function tries not to return a gym.spaces.Space

        Returns:
            gym.spaces.Space: Space after func has operated on it
        """
        tmp = EnvSpaceUtil.iterate_over_space_likes(func, (space, ), True, *func_args, **func_kwargs)
        if isinstance(tmp, gym.spaces.Space):
            return tmp
        raise RuntimeError("bug spat, should never happen, iterate_over_space tried to return something other than a gym Space")

    @staticmethod
    def iterate_over_space_with_sample(func, space: gym.spaces.Space, sample: sample_type, *func_args, **func_kwargs) -> sample_type:
        """
        utility tool for iterate over space and sample to help mypy handle the return type correctly
        and be a little easier to use

        Args:
            func (_type_): function to call on sample and all sub samples, will recieve
                            func_args and func_kwargs as arguments
            space (gym.spaces.Space): space to call func on
            sample: (sample_type): sample to call func on

        Raises:
            RuntimeError: if this function tries to not return a sample

        Returns:
            sample: space sample after func has operated on it
        """
        tmp = EnvSpaceUtil.iterate_over_space_likes(func, (space, sample), False, *func_args, **func_kwargs)
        if not isinstance(tmp, gym.spaces.Space):
            return tmp
        raise RuntimeError("bug spat, should never happen, iterate_over_space tried to return something other than a sample_type")

    # TODO: maybe use this function in more places? maybe not? it could be slower?
    @staticmethod
    def iterate_over_space_likes(
        func,
        space_likes: typing.Tuple[typing.Union[gym.spaces.Space, sample_type], ...],
        return_space: bool,
        *func_args,
        **func_kwargs,
    ) -> typing.Union[gym.spaces.Space, sample_type]:
        """
        Iterates over space_likes which are tuple, dicts or the gym equivalent.
        When it encounters an actual item that is not a container it calls the func method.
        put any args, or kwargs you want to give to func in the overall call and they will be forwarded

        Parameters
        ----------
        func:
            the function to apply
        space_likes: typing.Tuple[typing.Union[gym.spaces.Space, sample_type], ...]
            the spaces to iterate over. They must have the same keywords for dicts and number of items for tuples
        return_space: bool
            if true the containers will be gym space equivalents
        func_args
            the arguments to give to func
        func_kwargs
            the keyword arguments to give to func

        Returns
        -------
        The contained result by calling func and stuffing back into the tuples and dicts in the call
        if return_space=True the containers are gym spaces
        """

        first_space_like = space_likes[0]
        val = None
        if isinstance(first_space_like, (gym.spaces.Dict, dict, OrderedDict)):
            new_dict = OrderedDict()
            keys: typing.KeysView
            if isinstance(first_space_like, gym.spaces.Dict):
                keys = first_space_like.spaces.keys()
            else:
                keys = first_space_like.keys()

            for key in keys:
                # type ignoring this because mypy is really struggling with the gym.Dict vs dict differences
                new_space_likes = tuple(spacer[key] for spacer in space_likes)  # type: ignore[index]
                new_dict[key] = EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=new_space_likes,
                    return_space=return_space,
                    *func_args,
                    **func_kwargs,
                )
            val = gym.spaces.Dict(spaces=new_dict) if return_space else new_dict  # type: ignore
        elif isinstance(first_space_like, (gym.spaces.Tuple, tuple)):
            new_tuple = [
                EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=new_space_likes,
                    return_space=return_space,
                    *func_args,
                    **func_kwargs,
                ) for new_space_likes in zip(*space_likes)  # type: ignore
            ]
            val = (gym.spaces.Tuple(tuple(new_tuple)) if return_space else tuple(new_tuple))  # type: ignore
        elif isinstance(first_space_like, Repeated):
            # if space_likes is longer than 1 that means that return_space = False
            # if there is only the space that means we need to generate just the space
            # itself and can use the second path, however for the case where we have a sample
            # it comes in as a list and we must iterate over the entire repeated list and process it
            if len(space_likes) > 1:
                repeated_samples = space_likes[1]
                assert isinstance(repeated_samples, MutableSequence), \
                    f'repeated_samples must be MutableSequence, received {type(repeated_samples).__name__}'
                val = [  # type: ignore
                    EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                        func,
                        space_likes=(first_space_like.child_space, sample),
                        return_space=return_space,
                        *func_args,
                        **func_kwargs,
                    ) for sample in repeated_samples
                ]  # type: ignore
            else:
                new_child_space = EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=(first_space_like.child_space, ),
                    return_space=return_space,
                    *func_args,
                    **func_kwargs,
                )

                val = Repeated(child_space=new_child_space, max_len=first_space_like.max_len)  # type: ignore
        else:
            val = func(space_likes, *func_args, **func_kwargs)  # type: ignore
        return val  # type: ignore

    @staticmethod
    def box_scaler(
        space_likes: typing.Tuple[gym.spaces.Space, sample_type],
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This scales a box space to be between the out_min and out_max arguments

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space, sample_type]
            the first is the gym spade to determine the input min and max
            the second is the sample of this space to scale
        out_min: float
            the minimum of the output scaling
        out_max: float
            the maximum of the output scaling

        Returns
        -------
        sample_type:
            the scaled sample with min of out_min and max of out_max
        """
        (space_arg, space_sample_arg) = space_likes
        if isinstance(space_arg, gym.spaces.Box) and space_arg.is_bounded():
            val = space_sample_arg
            in_min = space_arg.low
            in_max = space_arg.high
            norm_value = (out_max - out_min) * (val - in_min) / (in_max - in_min) + out_min
            return norm_value.astype(np.float32)
        return copy.deepcopy(space_sample_arg)

    @staticmethod
    def scale_sample_from_space(
        space: gym.spaces.Space,
        space_sample: sample_type,
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This is a convenience wrapper for box_scaler

        Parameters
        ----------
        space: gym.spaces.Space
            the space to use for the input min and max
        space_sample: sample_type
            the space sample to scale
        out_min: float
            the minimum of the output scaling
        out_max: float
            the maximum of the output scaling

        Returns
        -------
        sample_type:
            the scaled sample with min of out_min and max of
            out_max (this is in dicts and tuples the same as space_sample was)
        """
        return EnvSpaceUtil.iterate_over_space_with_sample(
            func=EnvSpaceUtil.box_scaler,
            space=space,
            sample=space_sample,
            out_min=out_min,
            out_max=out_max,
        )

    @staticmethod
    def box_unscaler(
        space_likes: typing.Tuple[gym.spaces.Space, sample_type],
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        Unscales the space_sample according to be the scale of the input space.
        In this sense out_min and out_max are the min max of the sample

        Parameters
        ----------
        space_likes: typing.Tuple[gym.spaces.Space, sample_type]
            the first is the gym spade to determine the input min and max
            the second is the sample of this space to scale
        out_min: float
            the minimum of the sample
        out_max: float
            the maximum of the sample

        Returns
        -------
        space_type:
            the unscaled sample
        """
        (space_arg, space_sample_arg) = space_likes
        if isinstance(space_arg, gym.spaces.Box):
            norm_value = space_sample_arg
            assert isinstance(norm_value, np.ndarray), f'norm_value must be np.ndarray, received {type(norm_value).__name__}'
            in_min = space_arg.low
            in_max = space_arg.high
            val = (norm_value - out_min) * (in_max - in_min) / (out_max - out_min) + in_min
            return val
        return copy.deepcopy(space_sample_arg)

    @staticmethod
    def unscale_sample_from_space(
        space: gym.spaces.Space,
        space_sample: sample_type,
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This is a convenience wrapper for box_unscaler

        Parameters
        ----------
        space: gym.spaces.Space
            the gym space we will unscale to
        space_sample: sample_type
            the sample we want to unscale. thus this is a scaled version of the input space
            with a min,max defined by the arguments out_min,out_max
        out_min: float
            the minimum of the sample
        out_max: float
            the maximum of the sample

        Returns
        -------
        sample_type:
            the unscaled version of the space_sample. Thus the space should now contain this output sample
        """

        return EnvSpaceUtil.iterate_over_space_with_sample(
            func=EnvSpaceUtil.box_unscaler,
            space=space,
            sample=space_sample,
            out_min=out_min,
            out_max=out_max,
        )

    @staticmethod
    def convert_config_param_to_space(action_space: gym.spaces.Space, parameter: typing.Union[int, float, list, dict]) -> dict:
        """
        This is a a convert for parameters used in action space conversions

        Parameters
        ----------
        space: gym.spaces.Space
            the gym space that defines the actions
        parameter: typing.Union[int, float, list, dict]
            the parameter as defined in a a config

        Returns
        -------
        dict:
            actions are dicts, the output is a dict with the parameters set for each key
        """
        action_params = {}
        sample_action = action_space.sample().items()
        if isinstance(parameter, (int, float)):
            parameter = [parameter]  # change to list if needed
            if len(parameter) == 1:  # if length 1, broadcast to the proper length
                for key, value in sample_action:
                    action_params[key] = parameter * len(value)
        elif isinstance(parameter, list):
            if len(sample_action) != 1:
                raise ValueError(f"list configs can only be applied to action space dicts with single key: {sample_action}")
            for key, value in sample_action:
                if len(value) != len(parameter):
                    raise ValueError(f"config length does not match action length of {len(value)}. {parameter}")
                action_params[key] = parameter
        elif isinstance(parameter, dict):  # pylint: disable=too-many-nested-blocks
            for key, value in sample_action:
                if key not in parameter:
                    if isinstance(key, tuple):  # parameters may be specfied using tuple values as keys
                        action_params[key] = np.zeros(len(key))  # type: ignore[assignment]
                        for idx, sub_key in enumerate(key):
                            if sub_key not in parameter:
                                raise ValueError(f"action space key not in config key: {sub_key}, config: {parameter} ")
                            action_params[key][idx] = parameter[sub_key]
                    else:
                        raise ValueError(f"action space key not in config key: {key}, config: {parameter} ")
                elif len(value) != len(parameter[key]):
                    raise ValueError(f"config value length for key {key} does not match action length of {len(value)}. {parameter}")
                else:
                    action_params[key] = parameter[key]

        return action_params


def convert_gym_space(input_space: gym.Space, output_type: typing.Type[gym.Space]) -> gym.Space:
    """Converts a gym space to the specified output type.

    Parameters
    ----------
    input_space : gym.spaces
        The input space to convert.
    output_type : Type[gym.spaces.Space]
        The desired output space type.

    Returns
    -------
    gym.spaces.Space
        The converted gym space.

    Raises
    ------
    ValueError
        For Discrete output types, the input must contain only one discrete gym space.
    NotImplementedError
        Unsupported conversions.
    """

    if isinstance(input_space, gym.spaces.Dict):
        if output_type is SingleLayerDict:
            return SingleLayerDict(input_space)

        if output_type is gym.spaces.discrete.Discrete:
            flattened_space_list = space_utils.flatten_space(input_space)
            if len(flattened_space_list) > 1:
                raise ValueError(f'Too many inputs. Cannot convert: {input_space} into a single Discrete action')
            if isinstance(flattened_space_list[0], gym.spaces.discrete.Discrete):
                return flattened_space_list[0]
            raise ValueError(f'Conversion invalid: {flattened_space_list[0]} -> {output_type}')

    if isinstance(input_space, (gym.spaces.Dict, gym.spaces.MultiDiscrete, gym.spaces.Tuple, gym.spaces.MultiBinary)):
        if output_type is gym.spaces.box.Box:
            flattened_space = gym.spaces.flatten_space(input_space)
            # Check to make sure the resultant space is a box. Raise error if not.
            if not isinstance(flattened_space, gym.spaces.Box):
                raise NotImplementedError(f'{input_space} contains a space where the conversion is not implemented.')
            return flattened_space

    raise NotImplementedError(f'Conversion not implemented: {type(input_space)} -> {output_type}')


def convert_sample(sample: typing.Any, sample_space: gym.spaces.Space, output_space: gym.spaces.Space):
    """Converts the sample into the output_space format.

    Parameters
    ----------
    sample : typing.Any
        A sample from `sample_space`.
    sample_space : gym.spaces.Space
        The space associated with the sample.
    output_space : gym.spaces.Space
        The output space to convert the sample into.

    Returns
    -------
    typing.Any
        The converted sample.

    Raises
    ------
    NotImplementedError
        When an unsupported conversion is passed in.
    """
    # sample_space: Discrete -> output_space:Dict
    if isinstance(sample_space, gym.spaces.Discrete) and isinstance(output_space, gym.spaces.Dict):
        # Turn the discrete input value into a one-hot encoded vector
        sample = gym.spaces.flatten(sample_space, sample)
        return gym.spaces.unflatten(output_space, sample)

    # sample_space: Box -> output_space: Dict
    if isinstance(sample_space, gym.spaces.Box
                  ) and isinstance(output_space, (gym.spaces.Dict, gym.spaces.MultiDiscrete, gym.spaces.Tuple, gym.spaces.MultiBinary)):
        return gym.spaces.unflatten(output_space, sample)

    if isinstance(sample_space, (gym.spaces.Dict, gym.spaces.MultiDiscrete, gym.spaces.Tuple,
                                 gym.spaces.MultiBinary)) and isinstance(output_space, gym.spaces.Box):
        return gym.spaces.flatten(sample_space, sample)

    # sample_space: Dict -> output_space: SingleLayerDict
    if isinstance(sample_space, gym.spaces.Dict) and isinstance(output_space, SingleLayerDict):
        return SingleLayerDict.flatten_and_convert_to_ordered_dict(sample)

    # sample_space: SingleLayerDict -> output_space: Dict
    if isinstance(sample_space, SingleLayerDict) and isinstance(output_space, gym.spaces.Dict):
        return OrderedDict(flatten_dict.unflatten(sample, splitter='tuple'))

    if isinstance(sample_space, gym.spaces.Dict) and isinstance(output_space, gym.spaces.Discrete):
        flattened_sample_list = flatten_sample(sample)

        if len(flattened_sample_list) > 1:
            raise ValueError(f'Too many inputs. Cannot convert: {sample_space} into a single Discrete action')
        return flattened_sample_list[0]

    raise NotImplementedError(f'Conversion of sample is not implemented for: {sample_space}->{output_space}.')


class SingleLayerDict(gym.spaces.Dict):
    """A single layered gym.spaces.Dict

    Inherits gym.spaces.Dict

    Parameters
    ----------
    input_space_dict : gym.spaces.Dict
        Input gym space dictionary to flatten.
    """

    def __init__(self, input_space_dict: gym.spaces.Dict) -> None:
        converted_dict = SingleLayerDict._gym_dict_to_python_dict(copy.deepcopy(input_space_dict))
        converted_dict = SingleLayerDict.flatten_and_convert_to_ordered_dict(converted_dict.spaces)
        super().__init__(converted_dict)

    @staticmethod
    def _gym_dict_to_python_dict(input_space_dict: gym.spaces.Dict) -> typing.Any:
        """Recursively converts all nested gym.spaces.Dicts into python dictionaries

        Parameters
        ----------
        input_space_dict : gym.spaces.Space
            The input gym space dictionary to be converted into a python dictinoary.

        Returns
        -------
        Union[dict, gym.spaces.Dict]
            Converted output.

        """
        for key, value in input_space_dict.items():
            if isinstance(value, gym.spaces.Dict):
                # Converts the gym.spaces.Dict -> dictionary
                input_space_dict[key] = value.spaces
                SingleLayerDict._gym_dict_to_python_dict(input_space_dict[key])
        return input_space_dict

    @staticmethod
    def flatten_and_convert_to_ordered_dict(input_dict: typing.Union[dict, OrderedDict]) -> OrderedDict:
        """Flattens and sorts a python dictionary.

        Parameters
        ----------
        input_dict : typing.Union[dict, OrderedDict]
            A dictionary or OrderedDict.

        Returns
        -------
        OrderedDict
            An ordered dictionary sorted by the key.
        """
        return OrderedDict(list(flatten_dict.flatten(input_dict, reducer='tuple').items()))


def flatten_sample(sample):
    """Flattens a sampe into its primitive components using rllib flatten technique.

    Primitive components are any non Tuple/Dict types.

    Args:
        sample: action or observation to flatten. This may be any
            supported type (including nested Tuples and Dicts).

    Returns:
        List[prim_data]: The flattened list of primitive types. This list
            does not contain Tuples or Dicts anymore.
    """

    def _helper_flatten(sample_, return_list):

        if isinstance(sample_, tuple):
            for s in sample_:
                _helper_flatten(s, return_list)
        elif isinstance(sample_, (dict, OrderedDict)):
            for k in sample_:
                _helper_flatten(sample_[k], return_list)
        else:
            return_list.append(sample_)

    ret = []
    _helper_flatten(sample, ret)
    return ret
