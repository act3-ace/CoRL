# corl 1D Docking Example

### Intro
This document explains how to quickly get started working with the corl.
It will walk you through installation of the corl repo and dependencies
and how to launch a training loop for a simple environment: 1D Docking.

The code for the 1D Docking environment serves as a documented example of how
to interface with the corl framework to create a custom environment.

### Installation
First clone the corl git repository (https://git.act3-ace.com/act3-rl/corl) 
to your working directory. For example:
```commandline
git clone https://git.act3-ace.com/act3-rl/corl.git
```

Navigate to the root of the local repository and use pip to install corl dependencies with the following commands.
Note: setting up a project-specific environment is recommended before running these commands.
```commandline
pip install -e path/to/corl
pip install -r path/to/corl/requirements.txt
pip install tensorflow
```

### Training
To launch a training loop, the module /corl/train.py is used. This module
must be passed the necessary config files at launch.
From the root of the repository, execute the following command:

```commandline
python corl/train_rl.py --cfg config/experiments/docking_1d.yml --compute-platform local
```

The config path after the `--config` flag defines the task, while the three strings following
the `-ac` flag define the agent name, agent config path, and platform config path respectively.
Multiple `-ac` flags may be used to define multiagent environments.

