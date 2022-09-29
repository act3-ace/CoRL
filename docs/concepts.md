<!---
The purpose of this file is to provide a conceptual description of the topics covered.  It should include NO code syntax unless it is in
a "see also" link.  Furthermore, it should NOT include configuration syntax.
-->

## Experiment
An experiment contains the full description of all of the various elements of the machine learning task, including the environment, agents, platforms, and policies.
## Environment
The environment specifies the overall goal of the experiment along with the constraints and rewards that inform the solution to the environment.
## Agents and Platforms
A platform is something that exists within the environment, such as an aircraft.
If the platform is controlled by a trained policy, it is an agent.
## Done
A done condition is a termination condition for the episode that represents either the success or failure of the agent.
Most done conditions apply to a single agent; however, shared dones can affect all agents.
### Types of Dones
* World: Done conditions that represent the state of the external world and are applicable to all agents.  Common examples include geographic or altitude limits.
* Task: Done conditions that apply to a single agent and represent the unique success and failure conditions for this particular agent.  Common examples include position capture, rejoin success, or death shot down.
* Platform: Done conditions that represent the physical constraints of the platform.  Common examples include speed constraints.
* Shared: Done conditions that consolidate the overall state of all agents to determine the final state and done status of all agents.
## Glue
A glue is an observation processor that takes a measurement and converts it to a new observation.
## Reward
A reward is the the benefit received by the agent for the actions taken.
## Parts: Sensors and Controllers
Platforms contain multiple discrete elements, such as sensors and controllers.
A sensor provides the platform with raw observations of the environment.
A controller allows an agent to convert a numeric action into a control signal for the underlying simulation.
## Simulator
A simulator determines how the control signals provided from the agent's action update the state of the world and return an observation to the agent.
## Episode Parameter Provider
An episode parameter provider controls the episode initialization parameters used by the environment on each reset.
All initialization parameters are random distributions that can be sampled to provide the initialization of a particular episode of the environment.
These parameters can also have an attached updater, which allows the episode parameter provider to adjust the hyperparameters of parameter distributions.
## References
References allow multiple dones, glues, or rewards to reference the same parameter value.
All items configured to use the same reference utilize the same draw of the parameter distribution so that the exact value is identical.
## Policy
## Plugins
