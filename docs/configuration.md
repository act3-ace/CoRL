<!---
The purpose of this file is to provide a complete description of the configuration files and their syntax.
-->

## General Configuration
The general configuration file contains pointers to other parts of the configuration, along with the ability to override what is provided in those referenced configurations.
### Required Elements
The general configuration needs to specify the experiment and the configuration of various components.
The experiment is specified using the key `experiment_class` with a key that is the import path to the experiment.
The experiment must be a subclass of `BaseExperiment`.

For the `RllibExperiment`, the other components are specified with the following syntax:
```yaml
rllib_configs:
    default: <override array>
    <other config>: <override array>
ray_config: <override array>
env_config: <override array>
tune_config: <override array>
```

The `rllib_configs` includes the rllib configuration for various platform configurations.
Which element of the `rllib_configs` is selected depends upon the `compute-platform` provide at the command line.
### Overrides
The syntax of the override array is `[!include filename1, !include filename2, *anchor1, *anchor2]`.
The number of included files and number of anchors is arbitrary.
Later elements take precedence over earlier items.

The filenames refer to separate configuration files that need to follow the format for ray, rllib, tune, or environments as described below.

An anchor is specified as:
```yaml
arbitrary_name: &anchor_name
    <key>: <value>
```
The keys and values need to match the configuration for whatever the anchor represents.
## Ray
## Tune
## RLLIB
### Hyperparameters
### Custom Models
## Environment
The environment needs to specify the following items:
```yaml
plugin_paths: A list of import paths to load into the plugin library.
simulator: Described below
platforms: The name of the platforms to load from the plugin library.
episode_parameter_provider: Described as a common element below.
dones:
    world: A list of world dones, each which follows the syntax for functors, described below.
    task:
        agent1: A list of task dones for agent1, each which follows the syntax for functors, described below.
        agentN: The names of the agents come from the experiment
simulator_reset_parameters:  A mapping with parameters for the reset method of the simulator class
reference_store:  Described as a common element below.  The environment reference store can resolve references in both the environment and agent configuration files.
```
### Simulator
The simulator configuration includes both the class and the initialization parameters of that class.
The syntax is:
```yaml
type: Name of the class as registered within the plugin library.
config:
    param1_name: Value for initialization param1
    param2_name: Value for initialization param2
```
## Agent
The agent needs to specify the following items:
```yaml
agent: Import path to the agent class
config:
    parts: List of parts as described below.
    episode_parameter_provider: Described as a common element below.
    glues: List of glues, each of which follows the syntax for functors, described below.
    dones: List of platform dones, each of which follows the syntax for functors, described below.
    rewards: List of rewards, each of which follows the syntax for functors, described below.
    simulator_reset_parameters:  A mapping with parameters for the reset method of the simulator class
    reference_store: Described as a common element below.  The agent reference store can only resolve references in the agent configuration file.
```
### Parts
## Platforms
## Policies
## Experiment
## Common Elements
### Episode Parameter Providers
The episode parameter provider defines a class and the initialization parameters of that class:
```yaml
type: import path to the episode parameter provider
config:
    param1_name: Value for initialization param1
    param2_name: Value for initialization param2
```

In some cases the values of the initialization parameters for an episode parameter provider for an agent needs to know the name of that agent.
As the agent configuration file is agnostic to the agent name, that can not be directly encoded in the file.
Therefore, use the string `%%AGENT%%` wherever the agent name is needed.
The agent class will automatically replace this string with the agent name.
Note that the environment episode parameter provider cannot use this syntax.

### Parameters and Updaters
A parameter defines the class and initialization parameters of that class:
```yaml
type: import path to the parameter
config:
    units: The units of this parameter, or null if not relevant
    update: A mapping of hyperparameters of this parameter and the updater that controls it.
    <other initialization parameters>: Value of these parameters.
```

An updater defines how the hyperparameters of a parameter are modified.
The updater itself has the syntax:
```yaml
type: import path to the updater
config:
    param1_name: Value for initialization parameter 1
    param2_name: Value for initialization parameter 2
```

These updaters are combined into an updater mapping that also specifies the name of the hyperparameter that they update.
As an example is worth more than an explanation, consider how a `BoundStepUpdater` is attached to the value element of a `ConstantParameter`:
```yaml
type: corl.libraries.parameters.ConstantParameter
config:
    units: <As is appropriate>
    value: <The initial value of the parameter>
    update:
        value:  # Matches the name of the hyperparameter above
            type: BoundStepUpdater
            config:
                bound: <Desired bound>
                step: <Desired step>
                bound_type: <Desired "min" or "max">
```

Note that parameters with multiple hyperparameters can have multiple updaters.

### Functors
A functor has the following syntax:
```yaml
functor: Import path to the class held by the functor.  This can be a done, glue, or reward.
name: Optional name for the functor.  The default is usually functor.__name__; however, some functors modify this, such as `SensorBoundsCheckDone`.  If any of the config elements are parameters, this name must be globally unique.
config:
    param1_name: Value for initialization parameter 1 as a constant, ValueWithUnits, Parameter, or anything else as appropriate.
    param2_name: Value for initialization parameter 2 as a constant, ValueWithUnits, Parameter, or anything else as appropriate.
references:
    param3_name: Key in the reference store
    param4_name: Key in the reference store
```

The units of a configuration element can be specified as a `ValueWithUnits` using the syntax `{"value": value, "units": units}`.

The collection of configuration elements and references must specify all necessary initialization parameters needed to instantiate the functor.

Dones need to be a direct `Functor`.  Rewards can be a `Functor` or a `FunctorWrapper`.  Glues can be a `Functor`, `FunctorWrapper`, or `FunctorMultiWrapper`.

### Reference Store
The reference store is a mapping between reference names and the parameter to which they reference.
The values need to follow the format for a Parameter as described above, potentially with updaters.
