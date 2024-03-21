#!/bin/bash

# COPY over the target files from the internal source
cp ../../corl/corl/ . -r
cp ../../corl/config/ . -r
cp ../../corl/test/ . -r
cp ../../corl/docker . -r
cp ../../corl/pyproject.toml . -r
cp ../../corl/poetry.lock . -r
cp ../../corl/docker . -r
cp ../../corl/corl/ . -r
cp ../../corl/config/ . -r
cp ../../corl/test . -r
cp ../../corl/CHANGELOG.md . -r
cp ../../corl/README.md .
cp ../../corl/.pre-commit-config.yaml .
cp ../../corl/.gitignore .
cp ../../corl/.devcontainer .
cp ../../corl/scripts -r .

# Clean up the files!!!
python ../../corl/sanitize.py

# poetry lock
poetry install --sync

# verify test pass
pytest test