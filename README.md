**The following Site/Repository is currently under construction. We are still porting items and updating instructions for github site/CICD.**

# ACT3 RL Core

***Core act3 reinforcement learning library*** - The Core Reinforcement Learning library is intended to enable scalable deep reinforcement learning experimentation in a manner extensible to new simulations and new ways for the learning agents to interact with them. The hope is that this makes RL research easier by removing lock-in to particular simulations.

The work is released under the follow APRS approval.
- Initial release of CoRL - Part #1 -Approved on 2022-05-2024 12:08:51 - PA Approval # [AFRL-2022-2455]"

Documentation 
- https://act3-ace.github.io/CoRL/

## Install
### Install the source - Miniconda - local host:

- [Miniconda Install Instruction](https://docs.conda.io/en/latest/miniconda.html)

```bash
# Create a virtual environment to install/run code
conda create -n CoRL python==3.10.4 
# Activate the virtual environment
conda activate CoRL
# install poetry
pip install poetry==1.2.1
# Install the CoRL dependencies
poetry install
```

### How to install pip package

## Build

### How to build the wheel file

The following project supports building python packages via `Poetry`. 

```bash
# Create a virtual environment to install/run code
conda create -n CoRL python==3.10.4 
# Activate the virtual environment
conda activate CoRL
# install poetry
pip install poetry==1.2.1
# Build the CoRL package
poetry build
```

### How to build the documentations - Local

The follow project is setup to use [MKDOCS](https://www.mkdocs.org/) which is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file.

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
            - clong@toyon.com (ML Training)
            - joshua.blackburn@stresearch.com (ML Training / System Integration)
            - bstieber@toyon.com (ML Training)
            - madison.blake@shield.ai (ML Infrastructure and Evaluation)
            - sfierro@toyon.com (ML Training / System Integration)
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
