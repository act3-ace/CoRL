Initial release of CoRL - Part #1 -Approved on 2022-05-2024 12:08:51 - PA Approval # [AFRL-2022-2455]"

# ACT3 RL Core

***Core act3 reinforcement learning library*** - The Core Reinforcement Learning library is intended to enable scalable deep reinforcement learning experimentation in a manner extensible to new simulations and new ways for the learning agents to interact with them. The hope is that this makes RL research easier by removing lock-in to particular simulations.

## Install
### Install the source - Miniconda - local host:

```bash
# Create a virtual environment to install/run code
conda create -n CoRL poetry==1.2.1
# Activate the virtual environment
conda activate CoRL
# Install the CoRL dependencies
poetry install
```

### How to install pip package

## Build

### How to build the wheel file

The following project supports building python packages via `Poetry`. 

```bash
# Create a virtual environment to install/run code
conda create -n CoRL poetry==1.2.1
# Activate the virtual environment
conda activate CoRL
# Build the CoRL package
poetry build
```

### How to build the documentations

The following project support documentation via MkDocs

To build the documentation:
```
mkdocs build
```

To serve the documentation:
```
mkdocs serve
```

## How to build the Docker containers

The following project support development via Docker containers in VSCode and on the DoD HPC. This is not strictly required but does provide the mode conveniet way to get started. ***Note:*** fuller documentation is available in the documentation folder or online docs. 

- ***Setup the user env file:*** in code directory run the following script  --> `./scripts/setup_env_docker.sh`
- ***Build the Docker containers using compose:*** run the following command --> `docker-compose build`


## How to build the documentation locally

This repository is setup to use [MKDOCS](https://www.mkdocs.org/) which is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file. Start by reading the introductory tutorial, then check the User Guide for more information.

- ***Install Mkdocs Modules*** in container/virtual environment run the following command --> `pip install -U -r mkdocs-requirements.txt`
- ***Build Documentation:*** Inside Docker container run the following command --> `python -m  mkdocs build`
- ***Serve Documentation:*** Inside Docker container run one of the following commands --> 
    - `python -m mkdocs server`
    - `python -m mkdocs serve --no-livereload`

## Running base examples

```bash
python -m corl.train_rl --cfg config/experiments/cartpole_v1.yml
```

# Contributors

- AFRL Autonomy Capability Team (ACT3)
    - AFRL ACT3
        - terry.wilson.11@us.af.mil
        - bejamin.heiner@us.af.mil
        - karl.salva@us.af.mil
        - james.patrick@us.af.mil
        - Training Team
            - cameron.long@jacobs.com (ML Training)
            - joshua.blackburn@stresearch.com (ML Training)
            - bstieber@toyon.com (ML Training)
- AFRL Autonomy Capability Team (ACT3) Safe Autonomy (SA) Team
    - kerianne.hobbs@us.af.mil (AFRL ACT3)
    - terry.wilson.11@us.af.mil (AFRL ACT3)

#  Designation Indicator

- Controlled by: Air Force Research Laboratory (AFRL)
- Controlled by: AFRL Autonomy Capability Team (ACT3)
- LDC/Distribution Statement: DIST-A
- POCs:
    - terry.wilson.11@us.af.mil (AFRL ACT3)
    - bejamin.heiner@us.af.mil (AFRL ACT3)
    - kerianne.hobbs@us.af.mil (AFRL ACT3)
    
# Notices and Warnings
