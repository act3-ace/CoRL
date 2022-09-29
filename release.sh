#!/bin/bash -e
# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.

# This is a US Government Work not subject to copyright protection in the US.

# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

# script looks for a variable __version__ in a python file and updates its value of the version
# Matches the examples:
# __version__ = "0.0.0"
# __version__ = '0.0.0'

# use poetry version to update version to next release
NEXT_RELEASE="$1"
poetry version ${NEXT_RELEASE}


# use poetry export to generate a lock file for sync
poetry export -f requirements.txt -o temp.lock.txt --without-hashes --without-urls --with dev --with=lint --with=test --with=docs --with=profile
cat requirements.dep.txt | cut -d ';' -f1 > requirements.dep.txt
