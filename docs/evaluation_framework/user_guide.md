
# User Guide

## 1. Overview

This document provides a in-depth overview of the CoRL evaluation framework.

## 2. Purpose

The evaluation framework allows a user to evaluate reinforcement learning (RL) agents trained using CoRL.

## 3. Get Started

### 3.1 Prerequisites

- Ensure that you have the latest version of CoRL
    - The evaluation framework comes packaged inside CoRL
- All dependencies for CoRL are also needed dependencies for the evaluation framework
- Episode parameter providers must be utilized
    - If your setup does not use episode parameter providers, your setup is incompatible with the evaluation framework
- All simulator reset parameters and reference store parameters must be backed by a `corl.libraries.Parameter` type
    - They cannot be stand alone floats or scalar values

## 4. Concepts

### 4.1 Required and Optional Processes

Evaluation framework has 3 required processes, which must be run sequentially:

- Evaluation:
    - executes test roll outs
    - records agent steps
    - records trajectory data
- Generate metrics
    - requires that evaluation has been run successfully
    - retrieves and/or calculates relevant metrics from agent steps
    - retrieves and/or calculates relevant metrics from trajectory data
- Visualization
    - requires that generate metrics has been run successfully
    - creates graphs and other forms of visualization from metrics

> Note: Each of the processes above can be run manually.  Alternatively they can be run using the evaluation framework pipeline launcher which chains and executes all three processes

Evaluation framework has two optional processes:

- Storage
    - is only available when all of the required processes above have been run successfully
    - tracks raw output and metrics of each required process
- Pipeline launcher
    - wraps required and optional processes so they can be run back-to-back

### 4.2 Configuration Files for Required and Optional Processes

Each process above requires a configuration file.

> See the configuration section below for how to write the configuration files.

Here, we provide an overview of the configuration files needed for each process.

- Evaluation process config:
    - provides a configuration for 5 objects that together run the evaluation process
    - requires a test cases config, which describes the initial conditions from which the roll outs must start
- Generate metrics process config:
    - requires a metrics configuration file, which defines what metrics are to be computed and/or retrieved
    - requires an alert config, which defines thresholds for displaying certain alerts
- Visualization config:
    - describes what visualization will be run
    - provides any needed visualization parameters
- Storage config:
    - An optional process
    - configuration file providing a concrete implementation of the `corl.evaluation.utils.storage.Storage` interface as well as any arguments to initialize the implemented class
- Pipeline requires:
      - evaluation configuration file
      - generate metrics configuration file
        - metrics configuration file
        - alerts configuration
      - visualization configuration
      - storage configuration, if storage is utilized

## 5. Write Configuration Files

This section covers writing the following:

- Writing the configuration file for the evaluate process
- Writing the configuration file for the generate_metrics process
  - Writing the configuration file for metrics
  - Writing the configuration file for alerts
- Writing the configuration file for the visualize process

### 5.1 Write the Evaluation Process Configuration

This section provides an in-depth overview of how to write the configuration file for the evaluate process.

The launcher for the evaluation process has 5 main objects that need to be instantiated, and 1 argument:

The objects are:

- RLlibTrainer
    - engine of the evaluation framework, responsible for executing rollouts
- Plugins
    - captures objects and entities that allow the RllibTrainer to be configured for unique cases
- Teams:
    - keeps track of which agents are participating in an evaluation
- Task:
    - captures the RLlib config file that was used during training and extracts needed portions of information
- Pandas:
    - sets up test cases
    > It is given the name of Pandas, because internally the data is handled as a Pandas DataFrame

Argument:

- Recorders:
    - describes how the evaluation framework should save its pickle files and artifacts

The codeblock below illustrates the 5 objects and argument:

```python
parser = jsonargparse.ArgumentParser()
parser.add_argument(
    "--cfg",
    help="path to a json/yml file containing the running arguments",
    action=jsonargparse.ActionConfigFile,
)

# Construct teams
parser.add_class_arguments(Teams, "teams", instantiate=True)

# How to construct the generic task/environment
parser.add_class_arguments(Task, "task", instantiate=True)

# How to generate the test cases to be provided to the environment
# Different types of test case matrix representations will be added here
parser.add_class_arguments(Pandas, "test_cases.pandas", instantiate=True)

# How to construct the plugins needed by generic evaluation
parser.add_class_arguments(Plugins, "plugins", instantiate=True)

# How to construct the engine to run evaluation
# If more tasks types are added they will be appended here
parser.add_class_arguments(
    RllibTrainer,
    "engine.rllib",
    instantiate=True,
)

# Recorders to publish results to
parser.add_argument("recorders", type=typing.List[Folder])

if path is None:
    args = parser.parse_args()
    instantiate = parser.instantiate_classes(args)
else:
    args = parser.parse_path(path)
    instantiate = parser.instantiate_classes(args)

return instantiate, args
```

Filename: corl/evaluation/launchers/launch_evaluate.py

Keep in mind: the evaluation config is written to:

- **instantiate the above 5 objects**
- **supply values for the recorders argument**

#### 5.1.1 Configure the Teams Object

```yaml
teams:
    team_participant_map:
        blue:
        - platform_config: config_for_platforms_on_this_team.yml
          agents:
            -   name: blue0
                agent_config: config_for_agent.yml
                policy_config: policy_config.yml
                agent_loader:
                    class_path: corl.evaluation.loader.check_point_file.CheckpointFile
                    init_args:
                        checkpoint_filename: CHECKPOINT_LOCATION
                        policy_id: blue0
    
    participant_id_schema: "%team_name%%idx%"
```

In the codeblock above  `teams:` specifies that the following configuration applies to the teams object.

Everything that follows is the actual configuration for that object.

`blue:` in the example above indicates that the configuration is specifying a configuration for the blue team.

We can have as many teams as we wish, so we can add a red team and a green team, for example.

Suppose that we want to specify that we have only one agent in this team called `blue0`. We next have to specify 4 main pieces of information to instantiate this agent:

- name of the agent
    - denoted `name: blue0`
- agent_config:
    - configuration file for the agent, can be found in the experiment configuration file used at training time
- policy_config:
    - configuration file for the policy, can be found in the experiment configuration file used at training time
- agent_loader configuration:
    - specify the type and location of the trained checkpoint

<!-- markdownlint-disable MD036 -->
**Loading an Agent**

The evaluation framework provides several classes for loading an agent.

These classes can be found in the corl.evaluation.loader module of the evaluation framework.

The currently supported loading classes are as follows:

- corl.evaluation.loader.check_point_file.CheckPointFile
    - Class to load a RLlib checkpoint
    - Init Args:
      - checkpoint_filename : path to the RLlib checkpoint
      - policy_id: the policy id within the RLlib checkpoint to load
- corl.evaluation.loader.heuristic.Heuristic
    - Class to load a heuristic agent or scripted agent
    - Init args:
      - None
- corl.evaluation.loader.weight_file.WeightFile
    - Class to load a h5 file
    - Init args:
      - h5_file_path: h5_file_path to load agent from

In the example above, we want to use the blue0 policy id that we are loading from a RLlib checkpoint file.

**Specifying an Agent within a Team**

Lastly, denote the naming convention for agents in teams by specifying `participant_id_schema`

A concrete example from the 1d docking problem is provided below:

Filename: corl/config/evaluation/launch/eval_1d_docking.yml

```yaml
teams:
  team_participant_map:
    blue:
    - platform_config: config/tasks/docking_1d/docking1d_platform.yml
      agents:
      - name: blue0
        agent_config: config/tasks/docking_1d/docking1d_agent.yml
        policy_config: config/policy/ppo/default_config.yml
        agent_loader:
          class_path: corl.evaluation.loader.check_point_file.CheckpointFile
          init_args:
            checkpoint_filename: CHECKPOINT_LOCATION
            policy_id: blue0
    
    participant_id_schema: "%team_name%%idx%"
```

#### 5.1.2 Configure the Task Object

The evaluation framework uses the same task config that was used during training.

The general format for configuring the task object is as follows:

```yaml
task:
    config_yaml_file: config/tasks/<PROBLEM_NAME>/<PROBLEM_NAME>_task.yml
```

- Start with `task:`
    - Denotes that you are configuring the task object
- Supply the argument in the `config_yaml_file:` field
    - Use the path to same task config that was used during training of your agent

#### 5.1.3 Configure the Test Cases/Pandas Object

The general format for configuring the Pandas object is as follows:

```yaml
test_cases:
  pandas:
    data: test_cases.yml
    source_form: FILE_YAML_CONFIGURATION / FILE_CSV 
    randomize: False
```

- Start with `test_cases:`
    - Denotes that the Pandas object is being used
- Supply the following arguments as a YAML dictionary as shown above:
    - Data : the document detailing test cases
        - These can be either in CSV or YAML
    - Source_form : the source form of the test cases
        - Can be either a YAML or CSV file
        - specify either FILE_YAML_CONFIGURATION or FILE_CSV
    - Randomize : whether to randomize the test cases

**Writing test cases in a YAML file**

The general format for the test_cases.yml is as follows:

```yaml
episode_parameter_providers:
    environment:
        reference_store:
          parameter0:
              values: [1,2,3,4]
        simulator_reset:
          parameter1:
              values: [1,2,3,4]

    <agent_name>:
        reference_store:
          parameter0:
              values: [1,2,3,4]
        simulator_reset:
          parameter1:
              values: [1,2,3,4]
```

Write values for the simulator reset parameters and reference_store parameters that are found in environment and agent configuration files for your environment in the format shown above.

See the following possible setups:

- If there are simulator reset parameters only in the environment configuration, specify the simulator reset parameters under `environment:` as shown above
- If there are simulator reset parameters only in the agent configuration, specify the name of the agent in place of <agent_name> and the simulator reset parameters, as appropriate
    - If there are multiple agents, repeat the block starting with <agent_name> for each agent
- If there are both environment and agent configurations, follow the general structure illustrated above, providing details of the simulator reset parameters from the agent and environment configuration

**Writing test cases in a CSV file**

The CSV file configuration is useful when you want to fix the environment and agent parameters to certain values.

The general format for the test_cases.csv is as follows:

- In the first line of the .csv file, where the header belongs, list all simulator reset parameters and all reference store values as a comma separated list
  - Ensure there are no spaces between the comma and the start of a new parameter name
  - Simulator reset parameters and reference store values must be structured as follows:

    `<AGENT_NAME/environment>.<simulator_reset/reference_store>.<parameter_name>`

- After listing the header values, specify the values desired for each parameter, as a comma separated list  

An example, assuming the environment has a single reference_store parameter and the blue0 agent has a single simulator reset parameter is:

```csv
environment.reference_store.parameter1,blue0.simulator_reset.parameter1
0,2
```

#### 5.1.4 Configure the Plugins Object

The plugin object ensures that the evaluation framework stays general purpose. Denote plugins in the config by writing `plugins:`

##### 5.1.4.1 Platform Serialization Plugin

> The platform serialization plugin is required by the evaluation framework

It is denoted by `platform_serialization:` and needs a `class_path` argument.

The `class_path` argument needs to specify a serialization function, which must be a subclass of corl.evaluation.runners.section_factories.plugins.platform_serializer.PlatformSerializer

Rules to follow for setting the platform serialization plugin

1. Subclass `corl.evaluation.runners.section_factories.plugins.platform_serializer.PlatformSerializer`

2. Review the definition of a platform object as in your simulation, and write a function that takes that object and converts it to a python dictionary.

More information about writing custom platform serialization function can be found in the Customizing the Evaluation Framework section.

The platform serialization plugin is specified in the evaluation config as follows:

```yaml
plugins:
    platform_serialization:
        class_path: <PATH.TO.FUNCTION>

```

##### 5.1.4.2 Eval_Config_Update Plugin

This is an optional plugin for the evaluation framework. This plugin modifies the internally created RLlib config with any user needed modifications.

This plugin is specified in the evaluation config as follows:

```yaml
plugins:
    eval_config_update:
        - class_path_to_config_update
```

A eval_config_update_function must sublcass the `corl.evaluation.runners.section_factories.plugins.config_updater.ConfigUpdate` class.

A user can specify as many config updates as needed, and can specify them as a yaml list.

Information about writing a config_update function can be found in the Customizing the Evaluation Framework section.

#### 5.1.5 Configure the RLlibTrainer Engine

The RLlibTrainer is the main driver in the evaluation framework. The RLlibTrainer is configured by denoting the following keys in the config

``` yaml
engine:
    rllib:
```

> Note: `engine:` is followed by `rllib:`, which is indented twice

There are a total of 5 main arguments that can be supplied to the RLlibTrainer as a YAML dictionary:

- debug_mode
    - controls whether in debug mode or not
    - boolean value
    - default value: false
- workers
    - number of workers
      - if higher than 0 will perform parallel rollouts
    - integer value
    - default value:  0
- envs_per_worker
    - number of environments per workers
    - integer value
    - optional parameter
    - default value: None
- callbacks
    - a list of callbacks
    - a callback can be specified by listing the class path to the callback in a yaml list
    - if no additional callbacks need to be specified, beyond the default evaluation callback, set the callbacks to an empty list i.e. callbacks: []
- trainer_cls
    - the RLlib Trainer class to use
    - default value: PPO
- horizon
    - the value of horizon
    - integer value
    - default value: None
- explore
    - whether to explore or not
    - boolean value
    - default value: false

A concrete example from the 1d docking setup.

``` yaml
engine:
    rllib:
        callbacks: []
        workers: 0
```

#### 5.1.6 Argument: Recorders

The recorders are responsible for saving the results of an evaluation.

To specify the recorders, in the config write `recorders:`.

Underneath that as a yaml list you need to specify the class_path and init args for each save type.

Currently the following class is provided for saving purposes:

`corl.evaluation.recording.folder.Folder`

The following yaml block is specified for saving purposes:

```yaml
recorders:
- class_path: corl.evaluation.recording.folder.Folder
  init_args:
    dir: DIRECTORY_SAVE_LOCATION
    append_timestamp: False
```

### 5.2 Write Configuration File for the Generate_Metrics Process

The general format of configuration file to launch the Generate_Metrics process is:

```yaml
artifact_evaluation_outcome:
  location:
    class_path: corl.evaluation.recording.folder.FolderRecord
    init_args:
      absolute_path: <LOCATION OF EVALUATION DATA>

artifact_metrics:
  location: <DIRECTORY AT WHICH TO PLACE METRIC PKL>

metrics_config: <CONFIG CONTAINING METRICS TO COMPUTE>
alerts_config: <CONFIG CONTAINING ALERTS TO COMPUTE>
```

Above is the general format for the configuration needed to launch the generate_metrics process.

The `absolute_path` and `location` attribute of the configuration file are specified by the user.

The `metrics_config` and `alerts_config` are required inputs to this process beyond the setting of the data locations for this process.

The Generate_Metrics process has two required inputs

1. `metrics_config` describes which metrics/quantities are to be computed.

2. `alerts_config` describes which alerts are to be computed.

Details on how to write the  `metrics_config` and `alerts_config` are detailed in the next subsections.

#### 5.2.1 Write the Metrics Configuration File

The metrics configuration file is one of the required inputs to the Generate_Metrics process.

A metrics config is divided into two separate portions with metrics at the `world` and `agent` level.

Metrics such as EpisodeLength and WallTime(Sec) exist at the world level.

The intent of a metric determines whether it is at the `world` or `agent` level.

If metric information comes from the agent data stored in a EpisodeArtifact, it is the `agent` level. If metric information comes arbitrarily from components in the EpisodeArtifact, it is at the `world` level.

A general format to follow is:

```yaml
world:
  <array of metrics to apply to world>
  
agent: 
  __default__:
    <array of metrics to apply to an agent>
  AGENT_NAME:
    <array of metrics to apply to an agent of the name AGENT_NAME>
```

If only a `__default__` is given, or there is no match to a known AGENT_NAME, then the metrics in `__default__` will be loaded for the agent.

It is important to note that metrics under `__default__` are loaded for all agents, metrics under <AGENT_NAME> are loaded specifically for that agent , in addition to the metrics specified under `__default__`.

A concrete example from the 1D-Docking environment:

```yaml
world:
  -    name: WallTime(Sec)
    functor: corl.evaluation.metrics.generators.meta.runtime.Runtime
    config:
      description: calculated runtime of test case rollout
  -
    name: AverageWallTime
    functor: corl.evaluation.metrics.aggregators.average.Average
    config:
      description: calculated average wall time over all test case rollouts 
      metrics_to_use: WallTime(Sec)
      scope: null

  -
    name: EpisodeLength(Steps)
    functor: corl.evaluation.metrics.generators.meta.episode_length.EpisodeLength_Steps
    config:
      description: episode length of test case rollout in number of steps 

  -
    name: rate_of_runs_lt_5steps
    functor: corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate
    config:
      description: alert metric to see if any episode length is less than 5 steps
      metrics_to_use: EpisodeLength(Steps)
      scope:
          type: corl.evaluation.metrics.scopes.from_string
          config: {name: "evaluation"}
      condition:
        operator: <
        lhs: 5

agent:
  __default__:
    -
      name: Result
      functor: corl.evaluation.metrics.generators.dones.StatusCode
      config: 
        description: was docking performed successfully or not
        done_condition: DockingDoneFunction
        
    -
      name: Dones
      functor: corl.evaluation.metrics.generators.dones.DonesVec
      config:
        description: dones triggered at end of each rollout
    -
      name: TotalReward
      functor: corl.evaluation.metrics.generators.rewards.TotalReward
      config: 
        description: total reward calculated from test case rollout

    -
      name: CompletionRate
      functor: corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate
      config:
        description: out of the number of test case rollouts how many resulted in successful docking 
        metrics_to_use: Result
        scope: null
        condition:
          operator: ==
          lhs:
            functor: corl.dones.done_func_base.DoneStatusCodes
            config: {value: 1} # 1 is win
```

Writing a custom metric will be discussed in Section 6.3.

#### 5.2.2 Write an Alerts Config

Alerts are condition checks on metrics, which if met, will raise an alert of the given type.

Alerts to be checked for when processing an evaluation are configured using a yaml file similar to the metrics yaml file at the `world` and `agent` level.

```yaml
  name: Short Episodes
  metric: rate_of_runs_lt_5sec
  scope: evaluation
  thresholds:
    - type: error
      condition: 
        operator: ">"
        lhs: 0
```

`Alert` required fields:

- `name`: The name that will be reported if this alert is triggered
- `metric`: The name of the metric to check for
- `scope`: The scope which the metric is to be checked on. Currently "event", "evaluation" or "tournament" are supported
    - Corresponds to the scope of the metric being checked
    - Refer to section 6.3.2 for a discussion on scope levels [6.3.2. Metric Generators](#632-metric-generators)
- `thresholds`: List of thresholds to check for. Note that one alert can have escalating thresholds

A threshold, specifies a limit that is reached , and its related severity.

`Threshold` required fields:

- `type`: string indicating the type severity
    - If the threshold is met, the type will be saved
    - A user can specify any string, for this value, `error` is a pregistered value with predefined behavior
- `condition`: Condition to trigger the threshold

A condition is the expression that is to be evaluated to determine whether a threshold has been met.

`condition` required fields:

- `operator`: operator to apply. Currently supported ">", "<", and "=="
- `lhs`: value of left hand side of conditional

**Special Alert `type` behavior**

- "error": During processing if "raise-on-error-alert" flag is not disabled (true by default), then processing the scene will result in an exception being thrown

### 5.3 Write Configuration File for the Visualization Process

A general format is provided:

```yaml
artifact_metrics:
  location: <LOCATION OF METRICS FROM GENERATE_METRICS PROCESS>
  
artifact_visualization:
  location: <SAVE LOCATION FOR VISUALIZATIONS>

visualizations:
- class_path: <VISUALIZATION_CLASS_PATH>
  init_args:
    <INIT ARGS FOR VISUALIZATION CLASS>
```

The `artifact_metrics` block specifies the location of the data from the generate_metrics process.

The `artifact_visualization` block sets the location for where the visualization data will be stored.

The `visualizations` block accepts a yaml list of class paths and init args of the visualizations that are to be created. In order to register a visualization here it must be a subclass of : `corl.evaluation.visualization.visualization.Visualization`.

## 6. Customize the Evaluation Framework

### 6.1 Write the Platform Serialization Plugin

Every simulation/environment requires a platform serialization function for running evaluation. The platform serialization function is responsible for determining how platforms in a simulation are to be serialized.

To write a platform serialization function, do the following

1. Subclass `corl.evaluation.runners.section_factories.plugins.platforms_serializer.PlatformSerializer`

2. Implement the `serialize` method per the implementation of a platform in your environment.
   - The objective is to convert all properties of the object into a dictionary, for serialization purposes

Later specify the new platform_serialization function in the evaluation config as follows:

```yaml
plugins:
    platform_serialization:
        class_path: <PATH_TO_PLATFORM_SERIALIZATION_FUNCTION>
```

> Related: [5.1.4.1. Platform Serialization Plugin](#5141-platform-serialization-plugin)

### 6.2 Write the eval_config_update Plugin

In the event, there is a change that needs to be made to the internal Rllib config used by the RllibTrainer, implement a eval_config_update.

To implement a eval_config update do the following:  

1. Subclass the following:`corl.evaluation.runners.section_factories.plugins.config_updater.ConfigUpdate`
2. Implement the `update` method, as per your needs
3. Specify the config update in the configuration for the evaluate process as shown below

```yaml
plugins:
    eval_config_update:
        - class_path_to_config_update
```

> Refer to [5.1.4.2. Eval_Config_Update Plugin](#5142-eval_config_update-plugin)

### 6.3 Write Custom Metrics

Within the generate_metrics segment/process of the evaluation framework, it is possible that a user may want to extract a metric not already provided.

This section provides the details on how to implement a custom metric.

#### 6.3.1 Metrics

In the evaluation framework, there is a class called Metric, which refers to the type of a computed quantity.

Within the evaluation framework there are two main variants of the Metric class:

- TerminalMetric
    - contains only primitive datatypes
    - represent data in a raw form
    - think of these as floats, numbers, strings, and rates
- NonTerminalMetric
    - an instance of Metric, which itself contains other instances of Metric in it
    - example: the Vector NonTerminalMetric is a list of Metric

The abstraction of a Metric into these classes has been done to provide a consistent way to analyze and view data especially in the visualization segment of the evaluation framework

A Metric Generator and Metric Aggregator must return a value of type Metric.

All the TerminalMetrics provided in the evaluation framework can be found under: ```corl.evaluation.metrics.types.terminals```. 

All the NonTerminalMetrics provided in the evaluation framework can be found under: ```corl.evaluation.metrics.types.nonterminals```. 

If there is a type that you wish to use and is not currently provided, you may subclass either ```corl.evaluation.metric.TerminalMetric``` or ```corl.evaluation.metric.NonTerminalMetric```. Then according to the necessary specifications implement the metric type. 

#### 6.3.2 Metric Generators  

A metric generator is the application specific component of the evaluation framework that takes a record of an episode and generates a relevant metric.

A metric generator can have 3 different scopes:

- Event: A single instance of EpisodeData
- Evaluation: List of EpisodeData instances
- Tournament: A list of lists of EpisodeData instances

Currently the Event and Evaluation scopes are supported.

The general format for writing a metric generator is as follows:

```python
class XXXXXXX(MetricGeneratorXXXXXXScope):
    def generate_metric(self, params: EpisodeArtifact, **kwargs) -> Metric:
        # operate on params to extract the desired information, create and return metric.
        ...
```

A metric generator must obey the following rules:

- Must subclass either MetricGeneratorEventScope or MetricGeneratorEvaluationScope
    - To determine which, figure out whether the metric needs to be calculated over the entire evaluation or whether it can be calculated for a singular test case/event
- Must return a metric type

The evaluation framework comes with predefined MetricGenerators, they can be found under: ```corl.evaluation.metrics.generators```

#### 6.3.3 Metric Aggregator

A metric aggregator is a extension of a metric generator that computes a metric based on the notion that it is calculated based off another metric generator or aggregator. Here, rather than defining how to retrieve a metric, an operation is being defined: how to perform the aggregation of a certain metric.

An example can be seen at corl.evaluation.metrics.aggregators.accumulate.Accumulate

A metric aggregator must obey the following rule:

- Must subclass `corl.evaluation.metrics.generator.MetricGeneratorAggregator`

> Note that a metric aggregator is a type of metric generator; hence the name

The general format of a Metric Aggregator is as follows:

```python
class XXXXX(MetricGeneratorAggregator):
    def generate_metric(
        self, params: typing.Union[typing.List[Metric], Metric], **kwargs
    ) -> Metric:
        ...
```

The evaluation framework currently has the following implementations of metric aggregators:

- Accumulate
- Average
- Criteria Rate
- Sum

The definitions of the above can be found under: ```corl.evaluation.aggregators```.

### 6.4 Write Custom Callbacks

It is possible to specify additional callbacks to the evaluation framework.

All new callbacks must be a subclass of ray.rllib.algorithms.callbacks.DefaultCallbacks

After writing the callback the callback can be registered in the evaluation config as follows:

```yaml
engine:
    rllib:
        callbacks: 
            - class_path.callback1
            - class_path.callback2
```

> Note: if storing files over the course of an episode, store the file_path and filename in the episode.user_data dictionary specifically as follows:

```python
episode.user_data["episode_artifacts_filenames"]["name_of_file"] = file_path
```

### 6.5 Write a Custom Visualization

The evaluation framework supports custom visualizations.

To write a new visualization do the following:

1. subclass `corl.evaluation.visualization.visualization.Visualization`
2. Implement the `visualize` method
    - Keep a track of the following rules:
        - The `visualize` method utilizes the  the output of the generate_metrics process,the metrics pickle file, which contains a SceneProcessor, which contains information about what metrics were computed
        - The SceneProcessor data is accessed via the `self._post_processed_data` attribute
        - The output path for the visualizations is accessed via the `self.output_path` attribute
        - A visualization should store its output in the form of `self.output_path/<FILENAME>`  
3. As needed specify any needed values in the init method of the class, as these are configurable in the yaml file

### 6.6 Utilize Custom Storage

The evaluation framework supports customization of how and where evaluation results and artifacts can be saved.

To this end there exists a launch_storage process which allows a user to place the outputs of the evaluation framework into a storage utility of their choice.

To utilize this functionality a user must subclass the following class:
`corl.evaluation.util.storage.Storage`

The class must implement the load_artifacts_config and store methods. The load_artifacts_config is reponsible to determining which artifacts are to be stored. The store method is responsible for carrying out the necessary logic to do the storage.

It is recommended to use a validator to ensure all objects you are loading in via the load_artifacts_config is what you require for the current class.

## 7. Usage

Each segment is run using a launch script. They are found under corl.evaluation.launchers. There is a launch script per evaluation framework process.

An example launch command is provided for each process. However, make sure to replace the fictional config arguments with arguments that are appropriate for your environment.

### 7.1 Run Evaluation Process

```bash
python -m corl.evaluation.launchers.launch_evaluate --cfg evaluation_config.yml
```

#### 7.1.1 Configuration for Running the Evaluation Process

Replace evaluation_config.yml with the appropriate evaluation configuration file. The evaluation config is the config written in section 5.1.

Refer to [5.1 Write the Evaluation Process Configuration](#51-write-the-evaluation-process-configuration)

### 7.2 Run generate_metrics Process

``` bash
python -m corl.evaluation.launchers.launch_generate_metrics --cfg gen_metrics_cfg.yml
```

#### 7.2.1 Format of yaml Configuration File to Run generate_metrics Process

Refer to [5.2 Write Configuration File for the Generate_Metrics process.](#52-write-configuration-file-for-the-generate_metrics-process)

### 7.3 Run Visualization Process

```bash
python -m corl.evaluation.launchers.launch_visualize --cfg visualize_cfg.yml
```

#### 7.3.1 Configuration to Launch Visualize Process

The configuration to launch the visualization process is detailed in section 5.3

Refer to [5.3 Write Configuration file for the Visualization Process](#53-write-configuration-file-for-the-visualization-process)

### 7.4 Run Storage Process

```bash
python -m corl.evaluation.launchers.launch_storage --cfg storage_config.yml
```

#### 7.4.1 Configuration File to Run Storage Process

The evaluation framework supports saving the results of the evaluation framework process to various storage utilities. Refer to section 6.6 on how to write a custom storage utility.

The general format of configuration to launch the storage utility:

```yaml
storage_utility:
  - class_path: <STORAGE-UTILITY-CLASS-PATH>
    init_args:
      init_arg1: value 
      init_arg2: value 

artifacts_location_config: 
  artifact1: <ARTIFACT1_LOCATION>
  artifact2: <ARTIFACT2_LOCATION>
```

After the 'storage_utility' keyword you must specify as a yaml list the storage utilities you wish to use.

After the 'artifacts_location_config' specify the artifacts you wish to store from the evaluation, as a yaml dictionary.

### 7.5 Run Evaluation Pipeline

```bash
python -m corl.evaluation.launcher.launch_pipeline --cfg pipeline_config.yml
```

#### 7.5.1 Configuration for the Pipeline Launcher

```yaml
eval_config: <EVAL_CONFIG> # requires the config for the evaluation process
gen_metrics_config: <GEN_METRICS_CONFIG> # requires the config for the gen_metrics config 
visualize_config: <VISUALIZE_CONFIG> # requires the config for the visualize config 
storage_config: <STORAGE_CONFIG> # requires the config for the storage config , is optional
```

The above format is the format needed for the pipeline launcher. The pipeline launcher requires the configurations for each of the processes. However the pipeline will handle the outputs of each process so that the next process has access to the necessary data. The storage config is an optional config, if you do not wish to utilize any storage.

The following inputs and outputs are handled by the pipeline, to ensure that the processes run back to back.

1. Evaluation
    - None
2. Generate_metrics
    - Retrieves and Sets Evaluation data output location
    - Sets the location of metrics data file ( pickle file containing computed metrics)
3. Visualization
    - Location of metrics data file from the generate_metrics step
4. Storage (Optional)
    - If utilized,the artifacts from each of the processes above is tracked
    - Evaluation data
    - Metrics data file
    - All rendered visualizations

## 8. Contact

Author: Vardaan Gangal
