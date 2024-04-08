#!/bin/bash

#
# The following script is intended as a quick use item to pull items
# across to the public repositories.
#

#
# Move internal directories to current to replace with latest versions
#
internal_corl_path=$1
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

#
# Remove internal specific items which provide no value to the 
# github instance
#
rm -rf docker/*.crt
rm -rf docker/*.sh

#
# Cleanup all files to remove internal items such as repository 
# addresses and etc. 
#
python $1/sanitize.py

#
# Install the items need and make sure lock is good
#
poetry lock
poetry install --sync

#
# Run the workflow for github (includes pytest and etc)
#
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

./bin/act

rm -rf bin

# Generate corl pdf
pip install code_to_pdf
sudo apt update && sudo apt install wget
wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6.1-2/wkhtmltox_0.12.6.1-2.jammy_amd64.deb
sudo apt install -f ./wkhtmltox_0.12.6.1-2.jammy_amd64.deb
code_to_pdf --title CoRL-$(date +%F).pdf corl
rm ./wkhtmltox_0.12.6.1-2.jammy_amd64.deb
