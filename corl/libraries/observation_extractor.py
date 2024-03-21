"""
-------------------------------------------------------------------------------
The Autonomous Capabilities Team (ACT3) Deep Reinforcement Learning (D-RL) Environment

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
-------------------------------------------------------------------------------

Observation Extractor
"""
import typing


class ExtractorSet(typing.NamedTuple):
    """Class defining the set of extractors to pull information about a specific observation"""

    value: typing.Callable
    space: typing.Callable
    unit: typing.Callable


def ObservationExtractor(
    observation,
    fields: list[str],
    indices: int | list[int] | None = None,
):
    """This function extracts a metric produced by a specific glue in observation space of an agent

    ---- Arguments ----
    observation         - A dict of an agent's observations
    platforms           - The platforms the glue is observing, needed to compute the glue's prefix
    fields              - Fields the extractor will attempt to walk through
    indices            - These will be accessed after the fields have been accessed, allowing
                          users to reduce arrays to single values
    ---- Raises ----
    RuntimeError        - Thrown when the fields do not exists in the glue's measurements
    """
    if indices is None:
        indices = []
    if not isinstance(indices, list):
        indices = [indices]

    value = observation
    for field in fields:
        if (isinstance(field, str) and field not in value) or (isinstance(field, int) and len(value) < field):
            raise RuntimeError(
                f"The field {field} is not present in the observation, the requested fields were {fields}, value was {value}"
            )
        value = value[field]
    for index in indices:
        value = value[index]
    return value


def ObservationSpaceExtractor(
    observation_space,
    fields: list[str],
):
    """Extract the observation space from a glue

     ---- Arguments ----
    observation         - A dict of an agent's observations
    fields              - Fields the extractor will attempt to walk through

    ---- Raises ----
    RuntimeError        - Thrown when the field does not exists in the glue's obs space
    """
    space = observation_space
    for field in fields:
        if field not in space.spaces:
            raise RuntimeError(
                f"The field {field} is not present in the observation space,"
                f"the requested fields were {fields}, space is {observation_space}"
            )
        space = space[field]
    return space
