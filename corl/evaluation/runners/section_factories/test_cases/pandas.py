"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
Loads pandas test cases
"""
import dataclasses
import enum
import logging
import typing

import jsonargparse
import pandas as pd
import yaml
from numpy.random import PCG64, Generator

from corl.evaluation.eval_logger_name import EVAL_LOGGER_NAME
from corl.evaluation.runners.section_factories.test_cases import config_parser, test_case_generator


@jsonargparse.typing.final
@dataclasses.dataclass
class Pandas:
    """Configure a Pandas Test Case"""

    class SourceForm(str, enum.Enum):
        """Format of the file given to this class"""

        FILE_CSV = "FILE_CSV"
        FILE_YAML_CONFIGURATION = "FILE_YAML_CONFIGURATION"
        BINARY = "BINARY"

    data: str | pd.DataFrame
    source_form: SourceForm

    data_frame: pd.DataFrame = dataclasses.field(init=False)

    seed: int = dataclasses.field(default=12345678903141592653589793)
    samples: float | None = dataclasses.field(default=None)
    randomize: bool = dataclasses.field(default=True)

    def __post_init__(self):
        self._name = None

    def name(self) -> str:
        """Get the name of the pandas test case

        Returns:
            str -- Name of pandas file
        """
        return self._name

    def generate(self):  # noqa: PLR0915
        """Generate pandas test cases"""

        logger = logging.getLogger(EVAL_LOGGER_NAME)

        test_cases = None
        randomize_config = None

        ###################################################
        ## Generate the test case data

        if self.source_form == Pandas.SourceForm.FILE_CSV:
            self._name = self.data.split("/")[-1]
            with open(self.data, encoding="utf=8") as f:
                test_cases_data = pd.read_csv(f, low_memory=False)
            logger.info("Reading test cases from %s", self.data)
            test_cases = test_cases_data
        elif self.source_form == Pandas.SourceForm.FILE_YAML_CONFIGURATION:
            self._name = self.data.split("/")[-1]
            with open(self.data, encoding="utf=8") as f:
                data = f.read()

            # Get a dict of the configuration for the randomized test cases
            config = yaml.safe_load(data)

            # Generate the value_config and the randomize_config
            if "episode_parameter_providers" in config:
                flat_config = config_parser.flatten_config(config.pop("episode_parameter_providers"))
                value_config = config_parser.resolve_config_ranges(
                    typing.cast(
                        config_parser.RESOLVED_VARIABLE_STRUCTURE_VALUE, config_parser.extract_config_element(flat_config, element="value")
                    )
                )
                randomize_config = typing.cast(
                    config_parser.EXTRACTED_VARIABLE_STRUCTURE_RANDOMIZE,
                    config_parser.extract_config_element(flat_config, element="randomize"),
                )
            else:
                value_config = {}
                randomize_config = {}

            # Read test cases
            test_cases = test_case_generator.create_test_cases(value_config)

        elif self.source_form == Pandas.SourceForm.BINARY:
            self._name = "evaluation"  # Not sure what to call it when given a binary
            test_cases = self.data
        else:
            raise RuntimeError(f"Unknown source form {self.source_form}")

        # Check that our parsing worked
        if test_cases is None:
            raise RuntimeError('"test_cases" variable is set to None, this is a non op as it should be set in the if/else tree above')

        ###############################################
        ## Now that we have test_cases loaded from _somewhere_ process them further

        # Not sure what this does, apparently needed
        if "test_case" in test_cases.columns:
            test_cases = test_cases.set_index("test_case")
        else:
            test_cases.index.name = "test_case"

        # Log
        logger.info("Number of test cases: %s", len(test_cases))
        logger.info("Number of variables: %s", len(test_cases.columns))

        # Some sanity checks
        if len(test_cases) == 0:
            raise ValueError("Initial test cases have length zero")

        # Configure random numbers
        logger.info("Random seed: %s", self.seed)
        rng = Generator(PCG64(self.seed))

        # Sample
        if self.samples is not None:
            logger.info("Sampling test cases")
            test_cases = test_case_generator.sample(rng.bit_generator, test_cases, self.samples)
            if len(test_cases) == 0:
                raise ValueError("Sampling reduced test cases to zero")
            logger.info("Tests cases remaining: %s", len(test_cases))

        # Randomize
        if self.randomize:
            if not randomize_config:
                raise ValueError("Cannot randomize without randomization parameters within the configuration")

            logger.info("Randomizing test cases")
            logger.info("Randomizing the following variables: %s", list(randomize_config.keys()))

            test_cases = test_case_generator.randomize_test_cases(test_cases=test_cases, gen=rng, config=randomize_config)

        self.data_frame = test_cases
