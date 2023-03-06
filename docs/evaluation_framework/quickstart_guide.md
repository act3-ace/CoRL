# Quick Start Guide

## 1. Overview

This document provides a introduction to the evaluation framework. In this guide a tutorial is given using the 1d-docking environment provided in the CoRL Repo

## 2. Prerequisites

Ensure you have the latest version of CoRL . The evaluation framework is packaged inside CoRL.
Keep in mind you need a trained agent to use the evaluation framework.

## 3. Obtain an Agent Checkpoint

For the purposes of this guide, we will be using the 1d docking example which is provided as the base example in the CoRL repo.

### 3.1. Train the 1d Docking Agent

To evaluate an agent you must first obtain a trained agent, it does not need to be fully trained. A couple of iterations will be sufficient. Use the command below:

```bash
python -m corl.train_rl --cfg config/experiments/docking_1d.yml
```

Keep a track of the location where the agent checkpoint is saved. You will need it for the next step.

## 4. Run the Evaluation Framework

Running the evaluation framework consists of running three separate processes , Evaluate, Generate_Metrics and Visualize.

We will start by detailing how to run the evaluate process, which does the job of executing the rollouts.

## 5. Run the Evaluate Process

In order to run the Evaluate process, we need to write a configuration file.  
The evaluation configuration file for the 1d docking environment can be found at config/evaluation/launch/eval_1d_docking.yml . The configuration has been reproduced below for explanation purposes.
Read all sections down below, you DO NOT need to write/copy the below into your own file.

``` yaml
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
            checkpoint_filename: CHECKPOINT_LOCATION_FROM_STEP1
            policy_id: blue0

  participant_id_schema: "%team_name%%idx%"

task:
    config_yaml_file: config/tasks/docking_1d/docking1d_task.yml

test_cases:
  pandas:
    data: config/evaluation/test_cases_config/docking1d_tests.yml
    source_form: FILE_YAML_CONFIGURATION
    randomize: False

plugins:
    platform_serialization:
        class_path: corl.evaluation.serialize_platforms.serialize_Docking_1d    

engine:
    rllib:
        callbacks: []
        workers: 0

recorders:
- class_path: corl.evaluation.recording.folder.Folder
  init_args:
    dir: /opt/data/corl/docking1d/
    append_timestamp: False

```

### 5.1. Teams

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
            checkpoint_filename: CHECKPOINT_LOCATION_FROM_STEP1
            policy_id: blue0

  participant_id_schema: "%team_name%%idx%"
```

This section of the config refers to which agents are participating in an evaluation. To denote this, we mention which team is the agent from i.e. team 1 - blue , team 2 - red so on so forth. Since in the 1d docking problem there is only 1 team, the team will be on team blue. Since there is also only one agent, we denote the agent as blue0. Additionally, we supply configuration details such as the policy config used at training time, as well as the agent config that was used during training.

To load the agent, we specify what loader we want to use. To load the RLlib checkpoint we use the following class:

`corl.evaluation.loader.check_point_file.CheckpointFile`

Additionally, there are init_args for this CheckpointFile specifying the actual checkpoint location and the policy_id. Note that policy id is the id of the policy within the checkpoint, which also happens to be blue0.  

The participant_id_schema line tells us that how agents within a team will be denoted, such as blue0, red0 etc.

### 5.2. Task

``` yaml
task:
    config_yaml_file: config/tasks/docking_1d/docking1d_task.yml
```

In this portion of the config, we specify the task config yaml file used during training, as its is used during evaluation as well.

### 5.3. Test Cases

```yaml
test_cases:
  pandas:
    data: config/evaluation/test_cases_config/docking1d_tests.yml
    source_form: FILE_YAML_CONFIGURATION
    randomize: False
```

In this block, we denote the test cases or which are the initial conditions that we want to set our agent to and execute a rollout from.

Currently, the evaluation framework sets up the test cases internally as a pandas DataFrame, hence the 'pandas' keyword.

There are three main arguments here:

- data : the yaml file containing the test cases
- source_form : either FILE_CSV_CONFIGURATION or FILE_YAML_CONFIGURATION.
- randomize: should the test cases be randomized ?

#### 5.3.1. Writing Test Cases

```yaml
episode_parameter_providers:
  environment:
    simulator_reset:
      platforms:
        blue0:
          x: 
            value: [10,25,50]
          xdot: 
            value: 0
```

The file above can be found at config/evaluation/test_cases_config/docking1d_tests.yml

The values to randomize are listed within the simulator_reset parameters of the 1d_docking_env.yml file. Here, we set them to the value from which we want to evaluate them.

### 5.4. Plugins

``` yaml
plugins:
    platform_serialization:
        class_path: corl.evaluation.serialize_platforms.serialize_Docking_1d    
```

The evaluation framework has one required plugin namely platform_serialization. The evaluation framework needs to know how to serialize a platform in a simulation. To write this function look at the configuration and definition of a platform in that simulation and serialize the platform object as a dict.

The function to serialize a 1d docking platform can be found at: corl.evaluation.serialize_platforms.serialize_Docking_1d. For the sake of brevity we are not going into it here.

Refer to the user guide for more instructions on how to do platform serialization.

### 5.5. Engine

``` yaml
engine:
    rllib:
        callbacks: []
        workers: 0
```

Here, we configure the RLlibTrainer engine that actually executes the rollouts. We specify we have no other callbacks, (Ray RLlib Callbacks) and we are only going to run evaluation on a single node. Increasing the worker count allows running evaluation in parallel on more than 1 node.

### 5.6. Recorders

```yaml
recorders:
- class_path: corl.evaluation.recording.folder.Folder
  init_args:
    dir: /opt/data/corl/docking1d/
    append_timestamp: False
```

Here, we are denoting where to store the results of a an evaluation.

First, we specify that we want to save the results of an evaluation to a folder hence
'corl.evaluation.recording.folder.Folder'.

We pass init_args to specifying the save directory and whether or not we want to append a timestamp.

### 5.7. How to Setup the Configuration for Evaluation

As mentioned earlier, you do not need to write all the above into your own config file.

Since the configuration file to evaluate 1d docking already exists, we need to MODIFY the file:

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
            checkpoint_filename: CHECKPOINT_LOCATION_FROM_STEP_1
            policy_id: blue0

  participant_id_schema: "%team_name%%idx%"

task:
    config_yaml_file: config/tasks/docking_1d/docking1d_task.yml

test_cases:
  pandas:
    data: config/evaluation/test_cases_config/docking1d_tests.yml
    source_form: FILE_YAML_CONFIGURATION
    randomize: False

plugins:
    platform_serialization:
        class_path: corl.evaluation.serialize_platforms.serialize_Docking_1d    

engine:
    rllib:
        callbacks: []
        workers: 0

recorders:
- class_path: corl.evaluation.recording.folder.Folder
  init_args:
    dir: YOUR_PREFERRED_SAVE_LOCATION
    append_timestamp: False

```

Open up the file config 'config/evaluation/launch/eval_1d_docking.yml'. At the locations marked ```CHECKPOINT_LOCATION_FROM_STEP_1``` and `YOUR_PREFERRED_SAVE_LOCATION`, specify your trained checkpoint and the location at which you want to save the evaluation results to.

Keep track of ```YOUR_PREFERRED_SAVE_LOCATION```; you will need it for future steps.

### 5.8. How to run

To run the evaluate process of the evaluation framework, run the following command:

```bash
python -m corl.evaluation.launchers.launch_evaluate --cfg config/evaluation/launch/eval_1d_docking.yml
```

You will see a ray.start starting in your console or terminal window and a ray.shutdown when this portion of the evaluation framework finishes.

## 6. Run Generate_Metrics Process

The next step of running the evaluation framework consists of running the Generate_metrics. This step is simpler than steps 2 and 3.

At this point, the evaluation framework has outputted raw data of pickle files depicting details about what rewards, dones, and agent actions were recorded.

This step processes those raw data into useful metrics.

### 6.1. Metrics Configuration File

Running Generate_Metrics requires a yaml file configuration specifying which metrics need to be used during this step. For the 1d docking problem, a configuration file for metrics already exists. The file has been reproduced below for explanation purposes.

```yaml
world:
  -
    name: WallTime(Sec)
    functor: corl.evaluation.metrics.generators.meta.runtime.Runtime
    config: {}
  -
    name: AverageWallTime
    functor: corl.evaluation.metrics.aggregators.average.Average
    config:
      metrics_to_use: WallTime(Sec)
      scope: null

  -
    name: EpisodeLength(Steps)
    functor: corl.evaluation.metrics.generators.meta.episode_length.EpisodeLength_Steps
    config: {}

  -
    name: rate_of_runs_lt_5steps
    functor: corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate
    config:
      metrics_to_use: EpisodeLength(Steps)
      scope:
          type: corl.evaluation.metrics.scopes.from_string
          config: {name: "evaluation"}
      condition:
        operator: <
        lhs: 5

agent:
  __default__ :
    -
      name: Result
      functor: corl.evaluation.metrics.generators.dones.StatusCode
      config: {}
    -
      name: Dones
      functor: corl.evaluation.metrics.generators.dones.DonesVec
      config: {}
    -
      name: TotalReward
      functor: corl.evaluation.metrics.generators.rewards.TotalReward
      config: {}
    -
      name: CompletionRate
      functor: corl.evaluation.metrics.aggregators.criteria_rate.CriteriaRate
      config:
        metrics_to_use: Result
        scope: null
        condition:
          operator: ==
          lhs:
            functor: corl.dones.done_func_base.DoneStatusCodes
            config: {value: 1} # 1 is win
```

At the simplest level, a metric is a combination of a functor and its config. Metrics can be at the agent or world level.

### 6.2. Alerts Configuration File

Alerts config are used to help raise possible issues to the runner of the command on possible issues that could have occurred.

In the 1d docking example, we utilize the following configuration for alerts:

```yaml
world:
  -
    name: Short Episodes
    metric: rate_of_runs_lt_5steps
    scope: evaluation
    thresholds:
      - type: warning
        condition:
          operator: ">"
          lhs: 0

```

filename: config/evaluation/alerts/base_alerts.yml

### 6.3. How to Run

In order to run the Generate_Metrics process, we need to write a YAML file that captures all the arguments needed for the process.

```yaml
artifact_evaluation_outcome:
  location: 
    class_path: corl.evaluation.recording.folder.FolderRecord
    init_args:
      absolute_path: <SAVE-LOCATION>

artifact_metrics:
  location: <SAVE-LOCATION>

metrics_config: config/evaluation/metrics/1d_docking.yml
alerts_config: config/evaluation/alerts/base_alerts.yml
```

filename: config/evaluation/launch/metrics_gen_1d_docking.yml

The format of the file for running the Generate_Metrics has been reproduced above.

At the evaluate step, a folder would have been created containing the trajectory data at the end of the process. That directory is found at ```<YOUR_PREFERRED_SAVE_LOCATION>``` from the previous step.  

Specify the value of ```<YOUR_PREFERRED_SAVE_LOCATION>```for the value of ```<SAVE-LOCATION>``` in this configuration file.

To follow along with this tutorial, open the following file config/evaluation/launch/metrics_gen_1d_docking.yml. Fill in the value of ```<SAVE-LOCATION>``` and ```<YOUR_PREFERRED_SAVE_LOCATION>``` as per the instructions above.

The process is run using the following command at the command line/console window:

```bash
python -m corl.evaluation.launchers.launch_generate_metrics --cfg config/evaluation/launch/metrics_gen_1d_docking.yml
```

A pickle file would have been saved at the location of ```<SAVE_LOCATION>```/metrics.pkl. This is needed for the next step.

## 7. Run Visualize Process

This step takes the pickle file from the previous step and creates visualizations

To run the visualize process we need to make a configuration file that captures all the arguments needed for the process.

```yaml
artifact_metrics:
  location: <SAVE_LOCATION>
  
artifact_visualization:
  location: <SAVE_LOCATION>/visualizations/

visualizations:
- class_path: corl.evaluation.visualization.print.Print
  init_args:
    event_table_print: true
```

filename: config/evaluation/launch/visualize_1d_docking.yml

Here, we are specifying the locations of metrics pickle file containing the computed metrics and the save location of all the visualizations.

For <SAVE_LOCATION> under ```artifact_metrics``` specify the directory from the previous step to where the pickle file is saved.

For <SAVE_LOCATION> under ```artifact_visualization``` specify the directory where you want to save the visualizations.

Under ```visualizations``` we specify the visualizations we want to run.

To follow along with this tutorial, open the following file config/evaluation/launch/visualize_1d_docking.yml. Fill in the value of ```<SAVE-LOCATION>``` as per the instructions provided.  

### 7.1. How to Run

To run the process, use the following command:

```bash
bash 
python -m corl.evaluation.launchers.launch_visualize --cfg config/evaluation/launch/visualize_1d_docking.yml
```

## 8. Run All Processes End to End via Pipeline Launcher

The evaluation framework provides a utility to run all processes back to back without manually specifying all paths in between steps.

The evaluation framework provides the evaluation framework pipeline launcher.

To utilize the evaluation framework pipeline launcher a .yaml configuration needs to be created:

```yaml
eval_config: config/evaluation/launch/eval_1d_docking.yml
gen_metrics_config: config/evaluation/launch/metrics_gen_1d_docking.yml
visualize_config: config/evaluation/launch/visualize_1d_docking.yml
```

filename:config/evaluation/pipeline_1d_docking.yml

> Note: If you look in the file, there is a  ```storage_config:``` key, remove it from the configuration file to follow along. Information about that key is provided in the User-Guide.

In terms of paths that are handled by the evaluation framework pipeline launcher, all inputs and outputs of the Generate_Metrics and Visualization processes are handled. Keep in mind, however, values for the inputs and outputs of these processes must not be empty in the configuration files, they should be set to dummy paths such as '/path' or be left to be the directories written into the files when they were run manually. The pipeline launcher requires that the configuration file is fully complete.

### 8.1. How to run

To run the pipeline launcher process

```bash
bash 
python -m corl.evaluation.launchers.launch_pipeline --cfg config/evaluation/launch/pipeline_1d_docking.yml
```
