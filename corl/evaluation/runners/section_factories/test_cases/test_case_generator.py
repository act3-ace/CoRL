"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Generate test cases from configuration
"""
import itertools
import logging

import pandas as pd
from numpy.random import BitGenerator, Generator

from corl.evaluation.eval_logger_name import EVAL_LOGGER_NAME
from corl.evaluation.runners.section_factories.test_cases import config_parser


def create_test_cases(config: config_parser.RESOLVED_VARIABLE_STRUCTURE_VALUE) -> pd.DataFrame:
    """Create the Cartesian product of test cases from a configuration.

    Parameters
    ----------
    config : config_parser.RESOLVED_VARIABLE_STRUCTURE_VALUE
        Single level configuration mapping as if produced by `config_parser.resolve_config_ranges`.  The keys may contain "." to represent
        nesting levels.  Each value is a sequence of possible values for that variable.

    Returns
    -------
    pandas.DataFrame
        Collection of all test cases from the initial condition configuration.  The columns are the variables from the keys of the input
        dictionary.  Each row is a unique test case.
    """
    logger = logging.getLogger(EVAL_LOGGER_NAME)
    num_cases = 1
    for v in config.values():
        num_cases *= len(v)
    if num_cases > 1000000:
        logger.warning("Number of test cases: %s", num_cases)

    keys = sorted(config.keys())
    values = [config[key] for key in keys]

    return pd.DataFrame(itertools.product(*values), columns=keys)


def sample(bgen: BitGenerator, test_cases: pd.DataFrame, amount: int | float) -> pd.DataFrame:
    """Sample test cases by count or percentage.

    Parameters
    ----------
    bgen : numpy.random.BitGenerator
        Underlying random number generator
    test_cases : pandas.DataFrame
        Test cases to sample.  Rows are test cases.  Columns are variables.
    amount : Union[int, float]
        Count (if >= 1) or percentage (if < 1) of test cases to return

    Returns
    -------
    pandas.DataFrame
        Sampled test cases, with fewer rows according to amount provided.
    """

    return test_cases.sample(n=int(amount), random_state=bgen) if amount >= 1 else test_cases.sample(frac=amount, random_state=bgen)


def generate_random(gen: Generator, config: config_parser.RANDOMIZE, value: pd.Series, variable: str) -> pd.Series:
    """Generates random samples from the distribution parameters

    Parameters
    ----------
    gen : numpy.random.Generator
        Underlying random number generator
    config : Mapping[str, Union[str, float]]
        Configuration for the distribution.  The name of the distribution is the key `distribution`.  The remaining keys are dependent on
        the type of distribution.  Supported distributions with their parameters are:
          - uniform: Uniform distribution
              - width: The width of the distribution, centered at value
          - angular_uniform: Uniform distribution on a periodic domain, such as angles.
              - width: The width of the distribution, ignoring the branch cut
              - minimum: Branch cut location at the minimum edge of the periodicity.  For example, -180.
              - maximum: Branch cut location at the maximum edge of the periodicity.  For example, 180.
          - normal: Normal distribution
              - scale: The standard deviation
    value : pd.Series
        Current value of the variable about which to randomize.
    variable : str
        Name of the variable for error messages

    Returns
    -------
    value : pd.Series
        Updated value

    Raises
    ------
    ValueError
        Improper parameters for the selected distribution
        Unknown distribution
    """

    value = value.copy()

    distribution = str(config["distribution"]).lower()

    if distribution == "uniform":
        if set(config.keys()) != {"distribution", "width"}:
            raise ValueError(f"{variable} does not contain proper parameters for uniform distribution")

        extent = float(config["width"]) / 2
        value += gen.uniform(low=-extent, high=extent, size=len(value))

    elif distribution == "angular_uniform":
        if set(config.keys()) != {"distribution", "width", "min", "max"}:
            raise ValueError(f"{variable} does not contain proper parameters for angular uniform distribution")

        extent = float(config["width"]) / 2
        value += gen.uniform(low=-extent, high=extent, size=len(value))

        max_val = float(config["max"])
        min_val = float(config["min"])
        period = max_val - min_val

        while any(value > max_val):
            value.loc[value > max_val] -= period
        while any(value < min_val):
            value.loc[value < min_val] += period

    elif distribution == "normal":
        if set(config.keys()) != {"distribution", "scale"}:
            raise ValueError(f"{variable} does not contain proper parameters for normal distribution")

        value += gen.normal(loc=0, scale=float(config["scale"]), size=len(value))

    else:
        raise ValueError(f'{variable} contains unknown distribution {config["distribution"]}')

    return value


def randomize_test_cases(
    test_cases: pd.DataFrame, gen: Generator, config: config_parser.EXTRACTED_VARIABLE_STRUCTURE_RANDOMIZE
) -> pd.DataFrame:
    """Randomize the values contained within a collection of test cases.

    Parameters
    ----------
    test_cases : pandas.DataFrame
        Test cases over which to randomize.  Columns are variables.  Rows are samples.  This value is updated in-place.
    gen : numpy.random.Generator
        Underlying random number generator
    config : config_parser.EXTRACTED_VARIABLE_STRUCTURE_RANDOMIZE
        Arguments for the distribution of each variable.  The keys of the outer mapping are the variable names, all of which must be columns
        in test_cases.  The inner mapping are the arguments for the random distribution.  See `generate_random` for the format of the inner
        mapping.

    Returns
    -------
    pd.DataFrame
        Updated test cases.

    Raises
    ------
    ValueError
        Variable names in configuration is not present in the test case columns.
    """

    test_cases = test_cases.copy()

    if unknown_columns := set(config.keys()) - set(test_cases.columns):
        raise ValueError(f"Unknown columns: {unknown_columns}")

    for col, col_config in config.items():
        test_cases.loc[:, col] = test_cases.loc[:, col].astype(float)
        test_cases.loc[:, col] = generate_random(gen=gen, config=col_config, value=test_cases.loc[:, col], variable=col)

    return test_cases
