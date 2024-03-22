#!/bin/bash

#
# The following script is intended as a quick use item to pull items
# across to the public repositories.
#

# COPY START - Move internal directories

internal_corl_path=../../corl
strings=(
    corl/
    config/
    test/
    docker/
    .devcontainer/
    pyproject.toml
    poetry.lock
    docker
    CHANGELOG.md
    README.md
    .pre-commit-config.yaml
    .gitignore
    scripts
)
for i in "${strings[@]}"; do
    cp -r $internal_corl_path/$i .
done

# Remove internal specific items
rm -rf docker/*.crt
rm -rf docker/*.sh

# Clean up the files!!!
python ../../corl/sanitize.py

# Install the items need and make sure lock is good
poetry lock
poetry install --sync

# verify test pass
# pytest test

# Run the workflow tests for github
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

./bin/act

rm -rf bin