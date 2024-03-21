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
from collections.abc import MutableSequence

import flatten_dict
import gymnasium
import numpy as np
from ray.rllib.utils.spaces import space_utils
from ray.rllib.utils.spaces.repeated import Repeated

from corl.libraries.property import CorlRepeated
from corl.libraries.state_dict import StateDict
from corl.libraries.units import Quantity, corl_get_ureg


class EnvSpaceUtil:
    """ENV Space Util"""

    _logger = logging.getLogger("EnvSpaceUtil")
    sample_type = OrderedDict | dict | tuple | np.ndarray | list | Quantity

    @staticmethod
    def deep_sanity_check_space_sample(  # noqa: PLR0912
        space: gymnasium.spaces.Space,
        sample: sample_type,
        key_stack: str = "",
    ) -> None:
        """Ensure space sample is consistent with space. This will give a traceback of the exact space that failed

        Parameters
        ----------
        space: gymnasium.spaces.Dict
            the space we expect the sample to conform to
        sample: sample_type
            sample that we are checking if it belongs to the given space
        key_stack: str
            string of the keys we have used when getting to this current spot in the observation_space, observation
            this is used for recursive calls do not set in the initial call to this function
        """
        if isinstance(space, gymnasium.spaces.Dict):
            if not isinstance(sample, StateDict | OrderedDict | dict):
                raise ValueError(
                    f"space{key_stack}={space} was a gymnasium.spaces.Dict type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a StateDict, OrderedDict, or dict\n"
                )
            for key, value in sample.items():
                EnvSpaceUtil.deep_sanity_check_space_sample(space.spaces[key], value, key_stack=f"{key_stack}[{key}]")
        elif isinstance(space, gymnasium.spaces.Tuple):
            if not isinstance(sample, tuple):
                raise ValueError(
                    f"space{key_stack}={space} is a gymnasium.spaces.Tuple type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a tuple type"
                )
            for idx, value in enumerate(sample):
                EnvSpaceUtil.deep_sanity_check_space_sample(space.spaces[idx], value, key_stack=f"{key_stack}[{idx}]")
        elif isinstance(space, gymnasium.spaces.Discrete):
            if not isinstance(sample, int | np.integer | np.ndarray):
                raise ValueError(
                    f"space{key_stack}={space} is a gymnasium.spaces.Discrete type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not an int or np.integer or np.ndarray"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however space{key_stack} has space definition of {space} {space.n}"
                )
        elif isinstance(space, gymnasium.spaces.Box):
            if not isinstance(sample, np.ndarray | list | np.floating):
                raise ValueError(
                    f"space{key_stack}={space} is a gymnasium.spaces.Box type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.float type"
                )
            if not space.contains(sample):
                sample_dtype = getattr(sample, "dtype", None)
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however "
                    f"space{key_stack} has space definition of {space} {space.low} {space.high} "
                    f"space dtype is {space.dtype}, sample dtype is {sample_dtype}"
                )
        elif isinstance(space, gymnasium.spaces.MultiBinary):
            if not isinstance(sample, np.ndarray | list | np.integer):
                raise ValueError(
                    f"space{key_stack}={space} is a gymnasium.spaces.MultiBinary type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.integer type"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however space{key_stack} has space definition of {space} {space.n}"
                )
        elif isinstance(space, gymnasium.spaces.MultiDiscrete):
            if not isinstance(sample, np.ndarray | list | np.integer):
                raise ValueError(
                    f"space{key_stack}={space} is a gymnasium.spaces.MultiDiscrete type but\n"
                    f"sample{key_stack}={sample} \n"
                    f"is a {type(sample)} type and not a np.ndarray, list, or np.integer type"
                )
            if not space.contains(sample):
                raise ValueError(
                    f"sample{key_stack} has value of {sample} however space{key_stack} has space definition of {space} {space.nvec}"
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
        Merge two dictionaries that also can contain sub dictionaries. This function returns the second dict
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
    def scale_space(space: gymnasium.spaces.Space, scale: float) -> gymnasium.spaces.Space:
        """
        Multiplies the low and high properties of all the Boxes in the given gymnasium space by the scale input
        Parameters
        ----------
        space: gymnasium.spaces.Space
            the gymnasium space to scale the Boxes of
        scale: float
            what to multiply the Box low and high by
        Returns
        -------
        gymnasium.spaces.Space
            the scaled gymnasium space
        """
        # TODO: this copy probably doesn't actually work but I can dream
        val = copy.deepcopy(space)
        if isinstance(space, gymnasium.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.scale_space(value, scale=scale)
            val = gymnasium.spaces.Dict(spaces=new_dict)
        elif isinstance(space, gymnasium.spaces.Tuple):
            new_thing = [EnvSpaceUtil.scale_space(sp, scale=scale) for sp in space.spaces]
            val = gymnasium.spaces.Tuple(tuple(new_thing))
        elif isinstance(space, gymnasium.spaces.Box):
            scaled_box = gymnasium.spaces.Box(
                low=np.multiply(space.low, scale).astype(np.float32),
                high=np.multiply(space.high, scale).astype(np.float32),
                shape=space.shape,
                dtype=np.float32,
            )
            val = scaled_box

        return val

    @staticmethod
    def zero_mean_space(space: gymnasium.spaces.Space) -> gymnasium.spaces.Space:
        """
        Returns a space object where every Box instance has its low and high shifted to be zero mean
        Parameters
        ----------
        space: gymnasium.spaces.Space
            The gymnasium space to zero mean

        Returns
        -------
        gymnasium.spaces.Space
            A gymnasium space the same as the input but with the Box instances shifted
        """
        # TODO: this copy doesn't actually work but I can dream
        val = copy.deepcopy(space)
        if isinstance(space, gymnasium.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.zero_mean_space(value)
            val = gymnasium.spaces.Dict(spaces=new_dict)
        elif isinstance(space, gymnasium.spaces.Tuple):
            new_thing = [EnvSpaceUtil.zero_mean_space(sp) for sp in space.spaces]
            val = gymnasium.spaces.Tuple(tuple(new_thing))
        elif isinstance(space, gymnasium.spaces.Box):
            mean = (space.high + space.low) / 2
            zero_mean_box = gymnasium.spaces.Box(low=space.low - mean, high=space.high - mean, shape=space.shape, dtype=np.float32)
            val = zero_mean_box

        return val

    @staticmethod
    def space_box_min_maxer(
        space_likes: tuple[gymnasium.spaces.Space],
        out_min: float = -1.0,
        out_max: float = 1.0,
    ) -> gymnasium.spaces.Space:
        """
        Makes a gymnasium box to the out_min and out_max range

        Parameters
        ----------
        space_likes: typing.Tuple[gymnasium.spaces.Space]
            the gymnasium space to turn all boxes into the scaled space
        out_min: float
            the new low for the boxes
        out_max: float
            the new high for the boxes

        Returns
        -------
        gymnasium.spaces.Space:
            the new gymnasium spaces where all boxes have had their bounds changed
        """
        space_arg = space_likes[0]
        if isinstance(space_arg, gymnasium.spaces.Box) and space_arg.is_bounded():
            return gymnasium.spaces.Box(low=out_min, high=out_max, shape=space_arg.shape, dtype=np.float32)
        return copy.deepcopy(space_arg)

    @staticmethod
    def normalize_space(
        space: gymnasium.spaces.Space,
        out_min=-1,
        out_max=1,
    ) -> gymnasium.spaces.Space:
        """
        This is a convenience wrapper for box_scaler

        Parameters
        ----------
        space: gymnasium.spaces.Space
            the gymnasium space to turn all boxes into the scaled space
        out_min: float
            the new low for the boxes
        out_max: float
            the new high for the boxes

        Returns
        -------
        gymnasium.spaces.Space:
            the new gymnasium spaces where all boxes have had their bounds changed
        """
        return EnvSpaceUtil.iterate_over_space(
            func=EnvSpaceUtil.space_box_min_maxer,
            space=space,
            out_min=out_min,
            out_max=out_max,
        )

    @staticmethod
    def get_zero_sample_from_space(space: gymnasium.spaces.Space) -> sample_type:
        """
        Given a gymnasium space returns an instance of that space but instead of sampling from the
        gymnasium space, returns all zeros. If the space is not a Box and we cannot iterate over it
        then we will sample from it.

        Parameters
        ----------
        space: gymnasium.spaces.Space
            The gymnasium space to zero sample from

        Returns
        -------
        sample_type
            The instance of the gymnasium space but all Box spaces are sampled as zero
        """
        val = space.sample()
        if isinstance(space, gymnasium.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.get_zero_sample_from_space(value)
            val = new_dict
        elif isinstance(space, gymnasium.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.get_zero_sample_from_space(sp) for sp in space.spaces]
            val = tuple(new_tuple)
        elif isinstance(space, gymnasium.spaces.Box):
            val = np.zeros(shape=space.shape, dtype=np.float32)
        elif isinstance(space, gymnasium.spaces.Discrete):
            val = 0
        return val

    @staticmethod
    def get_mean_sample_from_space(space: gymnasium.spaces.Space) -> sample_type:
        """
        Given a gymnasium space returns an instance of that space but instead of sampling from the
        gymnasium space, returns all zeros. If the space is not a Box and we cannot iterate over it
        then we will sample from it.

        Parameters
        ----------
        space: gymnasium.spaces.Space
            The gymnasium space to zero sample from

        Returns
        -------
        sample_type
            The instance of the gymnasium space but all Box spaces are sampled as zero
        """
        val = space.sample()
        if isinstance(space, gymnasium.spaces.Dict):
            new_dict = OrderedDict()
            for key, value in space.spaces.items():
                new_dict[key] = EnvSpaceUtil.get_mean_sample_from_space(value)
            val = new_dict
        elif isinstance(space, gymnasium.spaces.Tuple):
            new_tuple = [EnvSpaceUtil.get_mean_sample_from_space(sp) for sp in space.spaces]
            val = tuple(new_tuple)
        elif isinstance(space, gymnasium.spaces.Box):
            val = (space.high + space.low) / 2.0
        return val

    @staticmethod
    def add_space_samples(
        space_template: gymnasium.spaces.Space,
        space_sample1: sample_type,
        space_sample2: sample_type,
    ) -> sample_type:
        """
        Adds together two instances of gymnasium spaces. This only adds the ndarray or list types (that were sampled from Box)
        If the object is not a ndarray or list then the value of space_sample1 is returned by default
        Parameters
        ----------
        space_template: gymnasium.spaces.Space
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
        if isinstance(space_template, gymnasium.spaces.Dict):
            assert isinstance(space_sample1, dict)
            if isinstance(space_sample2, dict):
                new_dict = OrderedDict()
                for key, space_value in space_template.spaces.items():
                    value1 = space_sample1[key]
                    value2 = space_sample2[key]
                    new_dict[key] = EnvSpaceUtil.add_space_samples(space_value, value1, value2)
                val = new_dict
            elif isinstance(space_sample2, Quantity):
                new_dict = OrderedDict()
                for key, space_value in space_template.spaces.items():
                    value1 = space_sample1[key]
                    new_dict[key] = EnvSpaceUtil.add_space_samples(space_value, value1, space_sample2)
                val = new_dict
            else:
                raise ValueError("space_sample2 needs to be a dict or a quantity")
        elif isinstance(space_template, gymnasium.spaces.Tuple):
            assert isinstance(space_sample1, tuple)
            assert isinstance(space_sample2, tuple)
            new_tuple = [EnvSpaceUtil.add_space_samples(*args) for args in zip(space_template, space_sample1, space_sample2)]
            val = tuple(new_tuple)
        elif isinstance(space_template, gymnasium.spaces.Box):
            if isinstance(space_sample1, np.ndarray):
                val = np.array(space_sample1 + space_sample2)
            elif isinstance(space_sample1, list):
                assert isinstance(space_sample2, list)
                val = [value1 + value2 for value1, value2 in zip(space_sample1, space_sample2)]
            elif isinstance(space_sample1, Quantity):
                assert isinstance(space_sample2, Quantity)
                val = space_sample1.m + space_sample2.m
                val = corl_get_ureg().Quantity(val, space_sample1.u)
        else:
            val = copy.deepcopy(space_sample1)
        return val

    @staticmethod
    def clip_space_sample_to_space(space_sample: sample_type, space: gymnasium.Space, is_wrap: bool = False) -> sample_type:
        """
        Clips a space instance to a given space. After this the space should contain the space instance

        Parameters
        ----------
        space_sample: sample_type
            the space instance we are going to clip
        space: gymnasium.spaces.Space
            the gymnasium space to clip the instance to.
        Returns
        -------
        sample_type
            the clipped space instance
        """
        val = space_sample
        if isinstance(space, Repeated):
            assert isinstance(val, list)
            new_obs = [EnvSpaceUtil.clip_space_sample_to_space(item, space.child_space, is_wrap=is_wrap) for item in val]

            val = new_obs
        if isinstance(space, gymnasium.spaces.Dict):
            assert isinstance(space_sample, dict)
            new_dict = OrderedDict()
            for key, space_value in space.spaces.items():
                space_sample_value = space_sample[key]
                new_dict[key] = EnvSpaceUtil.clip_space_sample_to_space(space_sample_value, space_value, is_wrap)
            val = new_dict
        elif isinstance(space, gymnasium.spaces.Tuple):
            assert isinstance(space_sample, tuple)
            new_tuple = [EnvSpaceUtil.clip_space_sample_to_space(siv, sv, is_wrap) for siv, sv in zip(space_sample, space)]
            val = tuple(new_tuple)
        elif isinstance(space, gymnasium.spaces.Box):
            assert isinstance(space_sample, Quantity)
            assert isinstance(val, Quantity)
            # A glue should always return
            val_m = val.m
            if is_wrap:
                # Takes care of the case where the controls wrap at the min/max value
                # Example: Range = 0-1, value = 1.1 ----> clipping puts to .1
                if val_m > space.high:
                    val_m = space.low + (val_m - space.high)
                elif val_m < space.low:
                    val_m = space.high - (space.low - val_m)
            else:
                # Takes care of the case where the controls saturate at the min/max value
                # Example: Range = 0-1, value = 1.1 ----> clipping puts to 1
                val_m = np.clip(a=val_m, a_min=space.low, a_max=space.high)
            val = corl_get_ureg().Quantity(val_m, val.u)
        return val

    @staticmethod
    def iterate_over_space(func, space: gymnasium.spaces.Space, *func_args, **func_kwargs) -> gymnasium.spaces.Space:
        """
        utility tool for iterate over space to help mypy handle the return type correctly
        and be a little easier to use

        Args:
            func (_type_): function to call on space and all sub space, will receive
                            func_args and func_kwargs as arguments
            space (gymnasium.spaces.Space): space to call func on

        Raises:
            RuntimeError: if this function tries not to return a gymnasium.spaces.Space

        Returns:
            gymnasium.spaces.Space: Space after func has operated on it
        """
        tmp = EnvSpaceUtil.iterate_over_space_likes(func, (space,), True, *func_args, **func_kwargs)
        if isinstance(tmp, gymnasium.spaces.Space):
            return tmp
        raise RuntimeError("bug spat, should never happen, iterate_over_space tried to return something other than a gymnasium Space")

    @staticmethod
    def iterate_over_space_with_sample(func, space: gymnasium.spaces.Space, sample: sample_type, *func_args, **func_kwargs) -> sample_type:
        """
        utility tool for iterate over space and sample to help mypy handle the return type correctly
        and be a little easier to use

        Args:
            func (_type_): function to call on sample and all sub samples, will receive
                            func_args and func_kwargs as arguments
            space (gymnasium.spaces.Space): space to call func on
            sample: (sample_type): sample to call func on

        Raises:
            RuntimeError: if this function tries to not return a sample

        Returns:
            sample: space sample after func has operated on it
        """
        tmp = EnvSpaceUtil.iterate_over_space_likes(func, (space, sample), False, *func_args, **func_kwargs)
        if not isinstance(tmp, gymnasium.spaces.Space):
            return tmp
        raise RuntimeError("bug spat, should never happen, iterate_over_space tried to return something other than a sample_type")

    # TODO: maybe use this function in more places? maybe not? it could be slower?
    @staticmethod
    def iterate_over_space_likes(
        func,
        space_likes: tuple[gymnasium.spaces.Space | sample_type, ...],
        return_space: bool,
        *func_args,
        **func_kwargs,
    ) -> gymnasium.spaces.Space | sample_type:
        """
        Iterates over space_likes which are tuple, dicts or the gymnasium equivalent.
        When it encounters an actual item that is not a container it calls the func method.
        put any args, or kwargs you want to give to func in the overall call and they will be forwarded

        Parameters
        ----------
        func:
            the function to apply
        space_likes: typing.Tuple[typing.Union[gymnasium.spaces.Space, sample_type], ...]
            the spaces to iterate over. They must have the same keywords for dicts and number of items for tuples
        return_space: bool
            if true the containers will be gymnasium space equivalents
        func_args
            the arguments to give to func
        func_kwargs
            the keyword arguments to give to func

        Returns
        -------
        The contained result by calling func and stuffing back into the tuples and dicts in the call
        if return_space=True the containers are gymnasium spaces
        """

        first_space_like = space_likes[0]
        val = None
        if isinstance(first_space_like, gymnasium.spaces.Dict | dict | OrderedDict):
            new_dict = OrderedDict()
            keys: typing.KeysView
            keys = first_space_like.spaces.keys() if isinstance(first_space_like, gymnasium.spaces.Dict) else first_space_like.keys()

            for key in keys:
                # type ignoring this because mypy is really struggling with the gymnasium.Dict vs dict differences
                new_space_likes = tuple(spacer[key] for spacer in space_likes)  # type: ignore[index]
                new_dict[key] = EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=new_space_likes,
                    return_space=return_space,
                    *func_args,  # noqa: B026
                    **func_kwargs,
                )
            val = gymnasium.spaces.Dict(spaces=new_dict) if return_space else new_dict  # type: ignore
        elif isinstance(first_space_like, gymnasium.spaces.Tuple | tuple):
            new_tuple = [
                EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=new_space_likes,
                    return_space=return_space,
                    *func_args,  # noqa: B026
                    **func_kwargs,
                )
                for new_space_likes in zip(*space_likes)  # type: ignore
            ]
            val = gymnasium.spaces.Tuple(tuple(new_tuple)) if return_space else tuple(new_tuple)  # type: ignore
        elif isinstance(first_space_like, Repeated):
            # if space_likes is longer than 1 that means that return_space = False
            # if there is only the space that means we need to generate just the space
            # itself and can use the second path, however for the case where we have a sample
            # it comes in as a list and we must iterate over the entire repeated list and process it
            if len(space_likes) > 1:
                repeated_samples = space_likes[1]
                assert isinstance(
                    repeated_samples, MutableSequence
                ), f"repeated_samples must be MutableSequence, received {type(repeated_samples).__name__}"
                val = [  # type: ignore
                    EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                        func,
                        space_likes=(first_space_like.child_space, sample),
                        return_space=return_space,
                        *func_args,  # noqa: B026
                        **func_kwargs,
                    )
                    for sample in repeated_samples
                ]
            else:
                new_child_space = EnvSpaceUtil.iterate_over_space_likes(  # type: ignore[misc]
                    func,
                    space_likes=(first_space_like.child_space,),
                    return_space=return_space,
                    *func_args,  # noqa: B026
                    **func_kwargs,
                )
                val = CorlRepeated(child_space=new_child_space, max_len=first_space_like.max_len)  # type: ignore
        else:
            val = func(space_likes, *func_args, **func_kwargs)
        return val  # type: ignore

    @staticmethod
    def box_scaler(
        space_likes: tuple[gymnasium.spaces.Space, sample_type],
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This scales a box space to be between the out_min and out_max arguments

        Parameters
        ----------
        space_likes: typing.Tuple[gymnasium.spaces.Space, sample_type]
            the first is the gymnasium space to determine the input min and max
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
        if isinstance(space_arg, gymnasium.spaces.Box) and space_arg.is_bounded():
            in_min = space_arg.low
            in_max = space_arg.high
            norm_value = (out_max - out_min) * (space_sample_arg - in_min) / (in_max - in_min) + out_min
            return norm_value.astype(np.float32)
        return copy.deepcopy(space_sample_arg)

    @staticmethod
    def scale_sample_from_space(
        space: gymnasium.spaces.Space,
        space_sample: sample_type,
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This is a convenience wrapper for box_scaler

        Parameters
        ----------
        space: gymnasium.spaces.Space
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
        space_likes: tuple[gymnasium.spaces.Space, sample_type],
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        Unscales the space_sample according to be the scale of the input space.
        In this sense out_min and out_max are the min max of the sample

        Parameters
        ----------
        space_likes: typing.Tuple[gymnasium.spaces.Space, sample_type]
            the first is the gymnasium spade to determine the input min and max
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
        if isinstance(space_arg, gymnasium.spaces.Box):
            norm_value = space_sample_arg
            assert isinstance(norm_value, np.ndarray), f"norm_value must be np.ndarray, received {type(norm_value).__name__}"
            in_min = space_arg.low
            in_max = space_arg.high
            return (norm_value - out_min) * (in_max - in_min) / (out_max - out_min) + in_min

        return copy.deepcopy(space_sample_arg)

    @staticmethod
    def unscale_sample_from_space(
        space: gymnasium.spaces.Space,
        space_sample: sample_type,
        out_min: float = -1,
        out_max: float = 1,
    ) -> sample_type:
        """
        This is a convenience wrapper for box_unscaler

        Parameters
        ----------
        space: gymnasium.spaces.Space
            the gymnasium space we will unscale to
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
    def convert_config_param_to_space(action_space: gymnasium.spaces.Space, parameter: int | (float | (list | dict))) -> dict:
        """
        This is a convert for parameters used in action space conversions

        Parameters
        ----------
        space: gymnasium.spaces.Space
            the gymnasium space that defines the actions
        parameter: typing.Union[int, float, list, dict]
            the parameter as defined in a a config

        Returns
        -------
        dict:
            actions are dicts, the output is a dict with the parameters set for each key
        """
        action_params = {}
        sample_action = action_space.sample().items()
        if isinstance(parameter, int | float):
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
        elif isinstance(parameter, dict):
            for key, value in sample_action:
                if key not in parameter:
                    if isinstance(key, tuple):  # parameters may be specified using tuple values as keys
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


def convert_gymnasium_space(input_space: gymnasium.Space, output_type: type[gymnasium.Space]) -> gymnasium.Space:
    """Converts a gymnasium space to the specified output type.

    Parameters
    ----------
    input_space : gymnasium.spaces
        The input space to convert.
    output_type : Type[gymnasium.spaces.Space]
        The desired output space type.

    Returns
    -------
    gymnasium.spaces.Space
        The converted gymnasium space.

    Raises
    ------
    ValueError
        For Discrete output types, the input must contain only one discrete gymnasium space.
    NotImplementedError
        Unsupported conversions.
    """

    if isinstance(input_space, gymnasium.spaces.Dict):
        if output_type is SingleLayerDict:
            return SingleLayerDict(input_space)

        if output_type is gymnasium.spaces.discrete.Discrete:
            flattened_space_list = space_utils.flatten_space(input_space)
            if len(flattened_space_list) > 1:
                raise ValueError(f"Too many inputs. Cannot convert: {input_space} into a single Discrete action")
            if isinstance(flattened_space_list[0], gymnasium.spaces.discrete.Discrete):
                return flattened_space_list[0]
            raise ValueError(f"Conversion invalid: {flattened_space_list[0]} -> {output_type}")

    if (
        isinstance(
            input_space, gymnasium.spaces.Dict | gymnasium.spaces.MultiDiscrete | gymnasium.spaces.Tuple | gymnasium.spaces.MultiBinary
        )
        and output_type is gymnasium.spaces.box.Box
    ):
        flattened_space = gymnasium.spaces.flatten_space(input_space)
        # Check to make sure the resultant space is a box. Raise error if not.
        if not isinstance(flattened_space, gymnasium.spaces.Box):
            raise NotImplementedError(f"{input_space} contains a space where the conversion is not implemented.")
        return flattened_space

    raise NotImplementedError(f"Conversion not implemented: {type(input_space)} -> {output_type}")


def convert_sample(sample: typing.Any, sample_space: gymnasium.spaces.Space, output_space: gymnasium.spaces.Space):
    """Converts the sample into the output_space format.

    Parameters
    ----------
    sample : typing.Any
        A sample from `sample_space`.
    sample_space : gymnasium.spaces.Space
        The space associated with the sample.
    output_space : gymnasium.spaces.Space
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
    if isinstance(sample_space, gymnasium.spaces.Discrete) and isinstance(output_space, gymnasium.spaces.Dict):
        # Turn the discrete input value into a one-hot encoded vector
        sample = gymnasium.spaces.flatten(sample_space, sample)
        return gymnasium.spaces.unflatten(output_space, sample)

    # sample_space: Box -> output_space: Dict
    if isinstance(sample_space, gymnasium.spaces.Box) and isinstance(
        output_space, gymnasium.spaces.Dict | gymnasium.spaces.MultiDiscrete | gymnasium.spaces.Tuple | gymnasium.spaces.MultiBinary
    ):
        return gymnasium.spaces.unflatten(output_space, sample)

    if isinstance(
        sample_space, gymnasium.spaces.Dict | gymnasium.spaces.MultiDiscrete | gymnasium.spaces.Tuple | gymnasium.spaces.MultiBinary
    ) and isinstance(output_space, gymnasium.spaces.Box):
        return gymnasium.spaces.flatten(sample_space, sample)

    # sample_space: Dict -> output_space: SingleLayerDict
    if isinstance(sample_space, gymnasium.spaces.Dict) and isinstance(output_space, SingleLayerDict):
        return SingleLayerDict.flatten_and_convert_to_ordered_dict(sample)

    # sample_space: SingleLayerDict -> output_space: Dict
    if isinstance(sample_space, SingleLayerDict) and isinstance(output_space, gymnasium.spaces.Dict):
        return OrderedDict(flatten_dict.unflatten(sample, splitter="tuple"))

    if isinstance(sample_space, gymnasium.spaces.Dict) and isinstance(output_space, gymnasium.spaces.Discrete):
        flattened_sample_list = flatten_sample(sample)

        if len(flattened_sample_list) > 1:
            raise ValueError(f"Too many inputs. Cannot convert: {sample_space} into a single Discrete action")
        return flattened_sample_list[0]

    raise NotImplementedError(f"Conversion of sample is not implemented for: {sample_space}->{output_space}.")


class SingleLayerDict(gymnasium.spaces.Dict):
    """A single layered gymnasium.spaces.Dict

    Inherits gymnasium.spaces.Dict

    Parameters
    ----------
    input_space_dict : gymnasium.spaces.Dict
        Input gymnasium space dictionary to flatten.
    """

    def __init__(self, input_space_dict: gymnasium.spaces.Dict) -> None:
        converted_dict = SingleLayerDict._gymnasium_dict_to_python_dict(copy.deepcopy(input_space_dict))
        converted_dict = SingleLayerDict.flatten_and_convert_to_ordered_dict(converted_dict.spaces)
        super().__init__(converted_dict)

    @staticmethod
    def _gymnasium_dict_to_python_dict(input_space_dict: gymnasium.spaces.Dict) -> typing.Any:
        """Recursively converts all nested gymnasium.spaces.Dicts into python dictionaries

        Parameters
        ----------
        input_space_dict : gymnasium.spaces.Space
            The input gymnasium space dictionary to be converted into a python dictinoary.

        Returns
        -------
        Union[dict, gymnasium.spaces.Dict]
            Converted output.

        """
        for key, value in input_space_dict.items():
            if isinstance(value, gymnasium.spaces.Dict):
                # Converts the gymnasium.spaces.Dict -> dictionary
                gymnasium_dict = gymnasium.spaces.Dict(value.spaces)
                SingleLayerDict._gymnasium_dict_to_python_dict(gymnasium_dict)
                input_space_dict[key] = gymnasium_dict
        return input_space_dict

    @staticmethod
    def flatten_and_convert_to_ordered_dict(input_dict: dict | OrderedDict) -> OrderedDict:
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
        return OrderedDict(list(flatten_dict.flatten(input_dict, reducer="tuple").items()))


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
        elif isinstance(sample_, dict | OrderedDict):
            for k in sample_:
                _helper_flatten(sample_[k], return_list)
        else:
            return_list.append(sample_)

    ret = []
    _helper_flatten(sample, ret)
    return ret
