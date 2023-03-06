**The following Site/Repository is currently under construction. We are still porting items and updating instructions for github site/CICD.**

# Autonomy Capability Team (ACT3) Home Page https://www.afrl.af.mil/ACT3/

The Air Force Research Laboratory’s (AFRL) Autonomy Capability Team (ACT3) is an AI Special Operations organization whose mission is to Operationalize AI at Scale for the Air Force. Commissioned by the AFRL Commander, ACT3 leverages an innovative ‘start-up’ business model as an alternative approach to the traditional AFRL Technical Directorate R&D model by combining the blue sky vision of an academic institution; the flexibility of an AI startup; and the discipline of a production development company. ACT3 integrates the world’s best under one roof. The goal of the ACT3 business model is to define the shortest path to successful transition of solutions using AFRL’s internal expertise and collaborations with the best academic and commercial AI researchers in the world. Successful implementation may mean new technology or new implementation of existing technology.

# ACT3 RL Core

***Core act3 reinforcement learning library*** - The Core Reinforcement Learning library is intended to enable scalable deep reinforcement learning experimentation in a manner extensible to new simulations and new ways for the learning agents to interact with them. The hope is that this makes RL research easier by removing lock-in to particular simulations.

The work is released under the follow APRS approval.

|    Date    |     Release Number     | Description                                                      |
| :--------: | :--------------------: | :--------------------------------------------------------------- |
| 2022-05-20 |     AFRL-2022-2455     | Initial release of [ACT3 CoRL](https://github.com/act3-ace/CoRL) |
| 2023-03-02 | APRS-RYZ-2023-01-00006 | Second release of [ACT3 CoRL](https://github.com/act3-ace/CoRL)  |

Related Publications:
- https://breakingdefense.com/2023/01/inside-the-special-f-16-the-air-force-is-using-to-test-out-ai/
- https://www.wpafb.af.mil/News/Article-Display/Article/3244878/afrl-aftc-collaborate-on-future-technology-via-weeklong-autonomy-summit/
- https://aerospaceamerica.aiaa.org/year-in-review/demonstrating-and-testing-artificial-intelligence-applications-in-aerospace/

Documentation 
- https://act3-ace.github.io/CoRL/

![image](https://user-images.githubusercontent.com/102970755/193952349-108c1acd-ce58-4908-a043-28c2a53c85fa.png)

- Framework Overview - Hyper configurable environment enabling rapid exploration and integration pathways
   - **A framework for developing highly-configurable environments and agents**
     - Develop core components in python
     - Configure experiments/agents in json/yml
     - Provides tooling to help validate configuration files and give useful feedback when files are misconfigured
   - **Designed with integration in mind**
   - **Dramatically reduce development time to put trained agents into an integration or using a different simulation** 
     - Can work with any training framework
     - Currently limited to Ray/RLLIB due to multi-agent requirement
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
- Episode Parameter Provider (EPP) - **Domain Randomization & Curriculum Learning at Environment, Platform, and Agent based on training**
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
- Simulator Class - **Extensible interface for transitioning between Dubins and other simulator backends**
  - Responsible for setting up the world for a agents to manipulate
    - Setting up and configuring the simulation 
    - Creating the simulation platforms
    - Placing those platforms in the world
  - Responsible for knowing how to advance the simulation when requested
    - The simulation returns a simulation state when reset or advanced that rewards or done conditions can use
    - This state contains at least both the time and the list of simulation platforms
    - Responsible for saving any information about the current training episode
      - Saving video/logs
- Simulator Platforms + parts - **Extensible base interface for parts to be added to planforms with an integration focus.**
  - Simulation platforms represent some object that can be manipulated in the simulation
    - Car/plane/robot/etc.
  - Have a config file to allow modes of configuration
  - Each platform has a set of parts attached to it
  - Parts take simulation specific code and wrap it in an interface that allows agents to read from and write to them
    - Parts do nothing unless a user configures a connection between the agent and a part using a glue (to be explained)
   - Parts could include things such as a throttle, a game button, a steering wheel, etc.
   - Parts are registered to a simulator using a string `Sensor_Throttle`, `Controller_Throttle`, etc.
- Glues - **Connecting layers to allow exposing observable state to rewards, termination/goal criteria, and agents**
  - A stateful functor
  - Responsible for producing actions and observations for the agent
  - May directly read/write to parts or other glues
  - Glues reading/writing to each other is called “wrapping”
  - Glues implement the composable and reusable behavior useful for developers
    - Common glues turn any sensor part into an obs and apply actions to any controller part
    - Wrapper glues can implement behaviors such as framestacking, delta actions
  - May not directly read from the simulation, only interface through parts
- Rewards, Dones (Goal & Termination) - **Composable functors common interface for sharing rewards and termination criteria in a stateful manner**
  - Composable state functors
  - Rewards generate the current step reward for the agent
  - Dones evaluate if the episode should stop on the current timestep
    - These done’s can be triggered for either success or failure
  - Both Done and Reward Functors can view the entire simulation state to reward agents
  - Done conditions typically add to the state when they trigger to signify what type of Done they are
    - WIN/LOSE/DRAW
    - Rewards are processed after Done conditions during anupdate, so rewards can read these labels
  - There can be an arbitrary number of reward or done functors for an agent
- Agent + Experiment Class
   - Agent Class
      - Responsible for holding all of the Done/Reward/Glue functors for a given agent
      - Can be many agent classes per platform
         - When one agent class on a platform reaches a done, all on that platform do
      - Different subclasses may process information in different ways or do different things
   - Experiment Class
      - Responsible for setting up an experiment and running it
      - Configures and creates the environment
      - Creates and configures the agent classes
      - Use of this class allows for any arbitrary RL training framework to be used as the backend for training
- CoRL Integration and Simulator Swapping
   - In CoRL all simulation specific components must be registered and retrieved from a plug-in library
   - As long as a specific simulator has all of the parts registered to it that an agent needs, CoRL can swap the simulator and parts out from under an agent seamlessly
   - As long as the parts for the two simulators have the same properties (in terms of sensed value bounds or controller inputs) there is no difference to the agent between the two and the regular environment can be used for integration
   - Besides integration this also allows for cross simulation evaluation or training of an agent to be resumed in another simulator


## Benifits
- **CoRL helps make RL environment development significantly easier**
- **CoRL provides hyper configurable environments/agents and experiments**
- **Instead of a new file every time a new observation is added, now just add a few lines of config**
- **Makes it possible to reuse glues/dones/rewards between different tasks if they are general**
- **Provides tools to use both domain randomization and curriculum learning through EPP**
- **An integration first focus means that integrating agents to the real world or different simulators is significantly easier**


## Install
### Install the source - Miniconda - local host:

- [Miniconda Install Instruction](https://docs.conda.io/en/latest/miniconda.html)

```bash
# Create a virtual environment to install/run code
conda create -n CoRL python==3.10.4 
# Activate the virtual environment
conda activate CoRL
# install poetry
pip install poetry
# Install the CoRL dependencies
poetry install
# Pre-commit setup
pre-commit install
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
pip install poetry
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

The following project support development via Docker containers in VSCode. This is not strictly required but does provide the mode conveniet way to get started. ***Note:*** fuller documentation is available in the documentation folder or online docs. 

- ***Setup the user env file:*** in code directory run the following script  --> `./scripts/setup_env_docker.sh`
- ***Build the Docker containers using compose:*** run the following command --> `docker-compose build`

## Running base examples

```bash
    python -m corl.train_rl --cfg config/experiments/cartpole_v1.yml
```

# Initial Contributors

Initial contributors include scientists and engineers associated with the [Air Force Research Laboratory (AFRL)](https://www.afrl.af.mil/), [Autonomy Capability Team 3 (ACT3)](https://www.afrl.af.mil/ACT3/), and the [Aerospace Systems Directorate (RQ)](https://www.afrl.af.mil/RQ/): 

- ACT3's Autonomous Air Combat Operations (AACO) Team 
  - Terry Wilson(PM/PI)
  - Karl Salva (System Integration)
  - James Patrick (Modeling & Simulation)
  - Benjamin Heiner (AI Behavior Training Lead)
  - Training Team
    - Cameron Long (ML Training)
    - Steve Fierro (ML Training / System Integration)
    - Brian Stieber (ML Training)
    - Joshua Blackburn (ML Training / System Integration)
    - Madison Blake (ML Infrastructure and Evaluation)
- ACT3
  - Kerianne Hobbs 
  - Jared Culbertson
  - Hamilton Clouse
  - Justin Merrick
  - Ian Cannon
  - Ian Leong
  - Vardaan Gangal
    
# Designation Indicator

- Controlled by: Air Force Research Laboratory (AFRL)
- Controlled by: AFRL Autonomy Capability Team (ACT3)
- LDC/Distribution Statement: DIST-A
- POCs:
    - terry.wilson.11@us.af.mil (AFRL ACT3)
    - bejamin.heiner@us.af.mil (AFRL ACT3)
    - kerianne.hobbs@us.af.mil (AFRL ACT3)
# Designation Indicator

- Controlled by: Air Force Research Laboratory (AFRL)
- Controlled by: AFRL Autonomy Capability Team (ACT3)
- LDC/Distribution Statement: DIST-A
- POCs:
    - terry.wilson.11@us.af.mil (AFRL ACT3)
    - bejamin.heiner@us.af.mil (AFRL ACT3)
    - kerianne.hobbs@us.af.mil (AFRL ACT3)
    
# Notices and Warnings
