**The following Site/Repository is currently under construction. We are still porting items and updating instructions for github site/CICD.**

# ACT3 RL Core

***Core act3 reinforcement learning library*** - The Core Reinforcement Learning library is intended to enable scalable deep reinforcement learning experimentation in a manner extensible to new simulations and new ways for the learning agents to interact with them. The hope is that this makes RL research easier by removing lock-in to particular simulations.

The work is released under the follow APRS approval.
- Initial release of CoRL - Part #1 -Approved on 2022-05-2024 12:08:51 - PA Approval # [AFRL-2022-2455]"

Documentation 
- https://act3-ace.github.io/CoRL/

![image](https://user-images.githubusercontent.com/102970755/193951075-be97a4ba-a3bc-49b3-b3fc-dec3bc11407c.png)

- Framework Overview - Hyper configurable environment enabling rapid exploration and integration pathways
   - **A framework for developing highly-configurable environments and agents**
     - Develop core components in python
     - Configure experiments/agents in json/yml
     - Provides tooling to help validate configuration files and give useful feedback when files are misconfigured
   - **Designed with integration in mind**
   - **Dramatically reduce development time to put trained agents into an integration or using a different simulation** 
     - Can work with any training framework
     - Currently limited to Ray/RLLIB due to multi-agent requirement
   - **Environment pre-written, users implement plugins**
     - Simulator
     - Platforms & Platform Parts
     - Glues
     - Rewards
     - Dones
- Validators - **Configuration guarantees for enabling validation of user configuration going into the major components** 
  - All major CoRL python components have a validator
  - Validators are python dataclasses implemented through the pydantic library
  - Validators check and validate user configuration arguments going into the major components
    - If a component successfully initializes, the validators guarantee the developer that the data listed in the validator is available to them
    - If a component doesn’t initialize, a nice helpful error message is automatically produced by pydantic
  - Adds a pseudo static typing to python classes
- Episode Parameter Provider (EPP) - Domain Randomization & Curriculum Learning at Environment, Platform, and Agent based on training
  - An important tool for RL environments is the ability to randomize as much as possible
    - Starting conditions / goal location / etc.
    - This leads to more general agents who are more robust to noise when solving a task
  - Another tool sometimes used in RL is curriculum learning (CL)
    - Starting from an easier problem and gradually making the environment match the required specifications can significantly speed up training
  - CoRL Agents and the environment all have an epp, which provides simulator or user defined parameters to be used during a specific episode
    - Simulator classes know what parameters they expect to setup an episode
    - Configuration parameters to the various functors can all be provided from an EPP
  - An EPP can also update parameters over the course of training
    - Make a goal parameter harder based on the agents win rate
    - Open the environment up to wider bounds once the agent initially starts to learn
- Simulator Class - Extensible interface for transitioning between Dubins and other simulator backends
  - Responsible for setting up the world for a agents to manipulate
    - Setting up and configuring the simulation 
    - Creating the simulation platforms
    - Placing those platforms in the world
  - Responsible for knowing how to advance the simulation when requested
    - The simulation returns a simulation state when reset or advanced that rewards or done conditions can use
    - This state contains at least both the time and the list of simulation platforms
    - Responsible for saving any information about the current training episode
      - Saving video/logs
- Simulator Platforms + parts - Extensible base interface for parts to be added to planforms with an integration focus.
...
...

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
