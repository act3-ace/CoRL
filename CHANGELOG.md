## 3.14.10

### Bug Fixes

* Move self out of trial_name_prefix closure

## 3.14.9

### Bug Fixes

* Resolve "Fix reading policy configs"

## 3.14.8

### Bug Fixes

* checkpoint validator ()
* updates for making the number of bench eps configurable ()
* updates to fix lock ()

## 3.14.7

### Bug Fixes

* Update breaking scipy dependency ()

## 3.14.6

### Bug Fixes

* **configs:** fix issues with docking evaluation ()
* **record:** fixed EpisodeArtifact loading to create expected object format ()

## 3.14.5

### Bug Fixes

* various eval fixes ()

## 3.14.4

### Bug Fixes

* mkdocs change ()
* remove badges, termynal ()

## 3.14.3

### Bug Fixes

* only copy if config dir exists ()

## 3.14.2

### Bug Fixes

* add_simultor_ref_count ()

## 3.14.1

### Bug Fixes

* fix_eval_test_cases ()

# 3.14.0

### Features

* add_walltime_artifact ()

## 3.13.3

### Bug Fixes

* **docs:** update badges macro script ()

## 3.13.2

### Bug Fixes

* remove badges for pages site ()

## 3.13.1

### Bug Fixes

* docs update with act3-pt template, revise qsg ()

# 3.13.0

### Features

* include configs from installed packages ()

## 3.12.1

### Bug Fixes

* lock tensorflow to 2.15 ()
* small commit to kick cicd ()

## 3.12.1

### Bug Fixes

* lock tensorflow to 2.15 ()

# 3.12.0

### Features

* Add algorithm runner ()

## 3.11.1

### Bug Fixes

* add corl_path yaml directive ()

# 3.11.0

### Features

* enable site_pkg config inclusion ()

## 3.10.4

### Bug Fixes

* re-add trial creator ()

## 3.10.3

### Bug Fixes

* move config install out of ci ()

## 3.10.2

### Bug Fixes

* air bugfix ()

## 3.10.1

### Bug Fixes

* push of small change accidentally not pushed for resolving done functors ()

# 3.10.0

### Features

* ray2.9 + pydantic 2 ()

## 3.9.4

### Bug Fixes

* updates for the configs ()

## 3.9.3

### Bug Fixes

* Python3.11 support ()

## 3.9.2

### Bug Fixes

* fix_eval_bugs ()

## 3.9.1

### Bug Fixes

* update to ray 2.8 ()

# 3.9.0

### Features

* implement unit registry creation move from experiment to environment ()

## 3.8.1

### Bug Fixes

* convert_evaluator_to_pydantic ()

# 3.8.0

### Features

* use poetry venv instead of root ()

## 3.7.3

### Bug Fixes

* add_save_metadata ()

## 3.7.2

### Bug Fixes

* fix_obs_indexing ()

## 3.7.1

### Bug Fixes

* updates to the eval for working with air --- still needs other fixes /... ()

# 3.7.0

### Features

* add validator to corl quantity ()

## 3.6.9

### Bug Fixes

* add corldirectory paths ()

## 3.6.8

### Bug Fixes

* add CorlFilePath class for better file path validation ()

## 3.6.7

### Bug Fixes

* fix_space_normalization ()

## 3.6.6

### Bug Fixes

* gc the deletion of tmp_env ()

## 3.6.5

### Bug Fixes

* updates for the auto documentation via the Export setup experiment ()

## 3.6.4

### Bug Fixes

* Updates to the spelling in the code base --- Fixes the export setup experiment. ()

## 3.6.3

### Bug Fixes

* updates to remove lines ()
* updates to remove lines ()

## 3.6.2

### Bug Fixes

* updates for the public release process. ()

## 3.6.1

### Bug Fixes

* delete tmp_env object again ()

# 3.6.0

### Bug Fixes

* Add "pass" to BaseAgent reset() ()
* Fix linting errors ()
* Remove pass from BaseAgent.reset() ()

### Features

* Add reset to Base Agent ()

## 3.5.10

### Bug Fixes

* Evaluate generic recorder ()

## 3.5.9

### Bug Fixes

* changes to help with more repeatability ()

## 3.5.8

### Bug Fixes

* fix environment_wrappers file spelling mistake ()

## 3.5.7

### Bug Fixes

* fix_pseudo_policies ()

## 3.5.6

### Bug Fixes

* attempt to remove the cwd from ray init ()

## 3.5.5

### Bug Fixes

* hot fix to add missing **init**.py file -- no impact to functionality ()

## 3.5.4

### Bug Fixes

* Revert local modifications ()

## 3.5.3

### Bug Fixes

* add new yml loading capabilities and fix bug related to dict patches ()

## 3.5.2

### Bug Fixes

* add unit validator ()

## 3.5.1

### Bug Fixes

* fix scalar values not being seen properly by Quantities ()

# 3.5.0

### Features

* Add NonTrainableBaseAgent ()

# 3.4.0

### Features

* replace pint with built-in corl library ()

## 3.3.11

### Bug Fixes

* Only set seed if provided ()

## 3.3.10

### Bug Fixes

* init_ray_method ()

## 3.3.9

### Bug Fixes

* bugfix create_unit_converted_prop() ()

## 3.3.8

### Bug Fixes

* New Pytorch base Image, CoRL, and Poetry ()

## 3.3.7

### Bug Fixes

* remove ito calls and cleanup the env on the experiment before we run ()

## 3.3.6

### Bug Fixes

* fix_corrupted_measurment_units ()

## 3.3.5

### Bug Fixes

* Allow Factory to build types that do not accept wrapped parameter ()

## 3.3.4

### Bug Fixes

* use CorlRepeated so is_np_flattenable works ()

## 3.3.3

### Bug Fixes

* fix observe_sensor not being able to read dict ssensor props ()

## 3.3.2

### Bug Fixes

* adds the cwd to the working dir of the ray remote actors ()

## 3.3.1

### Bug Fixes

* small general updates and inference support ()

# 3.3.0

### Features

* updates from initial downstream integrations ()

## 3.2.6

### Bug Fixes

* Add subclass hook for end of configuration ()

## 3.2.5

### Bug Fixes

* documentation/linting updates ()

## 3.2.4

### Bug Fixes

* mem store issues ()

## 3.2.3

### Bug Fixes

* docker container fixes to install corl ()
* update properties to add common helper functions and update six_dof_props to fix unit of gload ()

## 3.2.2

### Bug Fixes

* Resolve "migrate linting to ruff" ()

## 3.2.1

### Bug Fixes

* minor updates for supporting league play ()

# 3.2.0

### Bug Fixes

* dockerfile pulled from *** ()

### Features

* pytorch 2 in docker containers ()

## 3.1.3

### Bug Fixes

* corl 3.0 bug fixes based on down stream integrations ()

## 3.1.2

### Bug Fixes

* update to default not auto ()

## 3.1.1

### Bug Fixes

* dockerfile release pipeline fix? ()

# 3.1.0

### Features

* Train rl refactor ()

# 3.0.0

* Merge branch 'linting' into 'main' ()

### BREAKING CHANGES

* corl 3.0

See merge request /act3-rl/corl!34

## 2.14.1

### Bug Fixes

* **cicd:** remove deprecated cicd scripts, functionality has been moved to a job ()

# 2.14.0

### Features

* support repeated obs ()

## 2.13.11

### Bug Fixes

* act3 python package url ()
* add feature to allow environment to disable action history saving to save memory ()
* Allow mixed agent configs ()
* automate release turn of building tagged images ()
* flake8 ()
* isort ()
* pylint ()
* remove pip conf and move secrets to build sections ()
* remove redundancy in compose ()
* sdf/fix random action policy ()
* syntax error preventing y-axis extents from updating ()
* update poetry in compose ()
* use temp method to avoid default linting job ()
* yapf ()

## 2.13.10

### Bug Fixes

* update act3 python job ()

# 1.0.0 (2023-07-27)

### Bug Fixes

* too many lines of code in run_experiment() ()
* 1. update requirements, 2. Make dict mapping, 3. add requirement ()
* A number of changes related to output units ()
* *** 163v2 ()
* Ability to evaluate array action space ()
* Acedt integration with evaluation framework ()
* Add 2D support to observation_units() ()
* add back policy ()
* add base props ()
* Add BenchmarkExperiment ()
* add BoxProp2D and update nan_check ()
* add cicd ()
* Add data parameter checking for TabularParameterProvider ()
* Add debug flag ()
* add debug mode ()
* Add embedded properties fields ()
* Add episode length done automatically ()
* Add EPP checkpoint ()
* add flag on the wrong command ()
* add flag to disable auto rllib config setup ()
* Add force units ()
* add glue obs clipping functionality and add AT_LEAST DoneStatusCode tensorboard output ()
* add gputil ()
* Add help message to debug flag ()
* Add in missing init file ()
* add index url ()
* Add int ()
* Add int ()
* Add int to assert ()
* add missing **init**.py ()
* add missing commit and fix test ()
* add missing install command ()
* Add mkdocstrings)
* add more files ()
* Add observe sensor unit tests ()
* Add output_units validator ()
* add partofwhole units ()
* add path to install ()
* add poetry pyproject.toml and poetry.lock ()
* Add pyinstrument for BenchmarkExperiment ()
* add python job back ()
* add requirement.lock file to semantic assets ()
* add reward_wrapper base classes ()
* Add root validator to BoxProp ()
* add seed argument to create space ()
* Add string value support to ConstantParameter ()
* Add support for calling method on remote epp ()
* add trial_str_functor plugin and fix verbosity argument to train_rl which was broken ()
* add trial_str_functor plugin and fix verbosity argument to train_rl which was broken ()
* Add unit test for BenchmarkExperiment ()
* Add unit test for ObserveSensorRepeated ()
* Add Units and Space Definition to Default Callback ()
* Add variable logging ()
* Added 2D support to get_observation() ()
* added ability for EpisodeDoneReward to consolidate done conditions and added optional sanity check ()
* Added asserts ()
* Added asserts ()
* Added BoxProp min() and max(), ukpdated create_converted_space() ()
* Added check for properties key ()
* Added comments ()
* added disable_exclusivity_check call to base6dofplatform ()
* Added env end_episode_on_first_agent_done config option ()
* Added int to assert ()
* Added list of lists support ()
* added missing item to docs ()
* added obs_relative_controller_dict ()
* Added units class to list ()
* Added units GetStrFromUnit ()
* Added validation of consistency ()
* Added validator for BoxProp.unit ()
* Added validator for ObserveSensorValidator.output_units ()
* added wrappers for dones and rewards, updated base agent to support ()
* Adding environment to extract items from environment state and manipulates accumulate non-terminal metric ()
* adding plugin paths ()
* adds fallback code to create_training_observations so that agent's that... ()
* adds step time in seconds and steps to episode metadata ()
* ADR must succeed at bound ()
* All platforms dones ()
* Allow Evaluator to specify multiple rllib config updates ()
* Allow non-agent platforms ()
* allow rllib experiment to allow non trainable only runs ()
* APRS 2 Release ()
* Assert on flattened sampled_control ()
* attempt 2 at docker + poetry ()
* attempt to get torch version down to 1.20.0 ()
* auto rllib ()
* auto rllib ()
* Avoid double platform delete ()
* bad key name ()
* bad merge ()
* BaseTimeSensor ()
* BoundStepUpdater reverse bounded by original value ()
* **boxprob2d:** update observe sensor glue to handle BoxProp2D ()
* **boxprob2d:** update sensors)
* **boxprop2d:** added boxprop2d to handle NetHack env ()
* Bug fix in min()/max() ()
* Bug fixes and mypy fixes ()
* bugfix launch pipeline.py ()
* bugfix to disable_exclusivity ()
* bump the code-server version and add tags for the GPUs (non specific to nvidia ()
* **callbacks:** cumulative reward per reward source added to custom metrics ()
* **callbacks:** match default callbacks to new signature ()
* Change agent removed action to random sample ()
* change all parts to accept platform, config, prop ()
* Change automatic name for SensorBoundsCheckDone ()
* change default value for skip_win_lose_sanity_check ()
* Change from numbers.Real to float ()
* Change functor from iterator to sequence. ()
* change memray to only install on linux ()
* change order of no-debug flag to fix default settings ()
* change paths and name to corl ()
* change to individual dep images ()
* change to nvidia-pytorch ()
* Change type ()
* Change velocity ned property to mpstas ()
* change wrap_dict to wrapped ()
* Changed argument names ()
* Changed flatten of keys ()
* changed return type to float ()
* changed typing ()
* Changed typing of output_units ()
* Changes to support deleting platforms from simulation ()
* check array shape ()
* clean up 6dof after merge ()
* clean up base properties ()
* clean up linting ()
* clean up linting ()
* cleaned up scripted action policy ()
* cleanup to fix repeated field convert ()
* cleanup to fix repeated field convert ()
* cleanup to remove references to items ()
* cleanup to remove references to items ()
* cleanup to remove references to items ()
* cleanup-eval-initialization ()
* Combine output unit validation ()
* Config cleanup for public release ()
* **configs:** change default configs to ignore_reinit_error True ()
* configurable state pickle writing on obs fail ()
* Connect EPP metrics in callback ()
* Consolidate test sensors ()
* Consolidate test sensors ()
* **controller:** continuous gym cntrllers work now ()
* correct the eval pipeline to use correct paths ()
* Corrected FuelProp unit from percent to fraction ()
* corrected the implementation of StatusCode and updated 1d docking config ()
* Create environment on driver in rllib experiment ()
* Create MachSpeed units ()
* Created MachSpeed units ()
* custom policy interface changes to support inference ()
* **dca:** added a unique_name field to TargetValueDifference. ()
* default exclusiveness to None ()
* default exclusiveness to None ()
* Default task dones in environment variable store ()
* deprecate agent_name == platform_name ()
* do not return empty obs for platforms without any glues ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** missed a couple changes ()
* **Dockerfile:** remove ray custom install ()
* **Dockerfile:** replace chmod ()
* **Dockerfile:** rest permissions on /tmp ()
* **Dockerfile:** rest permissions on /tmp ()
* **Dockerfile:** update DOCKER_OCI_REG ()
* **dockerfile:** update pip install ()
* **Dockerfile:** update registry arg ()
* **DOckerfile:** update versions on external deps ()
* **Dockerfile:** updates for 620 ()
* Don't call EPP.get_params from init ()
* Don't fail on redundant functor names if no parameters ()
* **dones:** added shared done base class ()
* dtype default ()
* Edge case consideration ()
* edit install poetry command ()
* Eliminate delete before use with done platforms ()
* Enable agent replacement for agent EPP configuration ()
* enable different parts and glue on each platform in multiplatform agent ()
* enable duplicate parts via unique part names ()
* Enable part references, functor children ()
* Ensure functor wrappers manage parameters correctly ()
* Ensure noop controller uses proper API ()
* EPP.get_params validation ()
* Error checking, horizon from rllib_config ()
* Error handling in epp validator ()
* eval default callbacks fix ()
* Eval episode artifact ()
* Eval Fixes ()
* Eval framework - Namespace TypeError fix for visualization launcher ()
* eval hotfix ()
* Evaluate supports "explore: False" ()
* Evaluation framework can output multiple visualizations ()
* Exclusiveness as property ()
* exception text ()
* Fix annotation state alignment ()
* fix BaseModel derivations ()
* Fix BenchmarkExperiment for new interface ()
* fix bug in obs_relative_delta_controller when dealing with multiple actions in a space ()
* fix class name ()
* fix commander pong eval and bugs in iterate_test_cases ()
* fix frozen weight loading not working, and allow target value diff to ignore invalid parts ()
* fix FunctorDictWrapper mangling configs of regular wrapper glues ()
* fix FunctorDictWrapper validator from crashing on non dict inputs ()
* Fix handling of platform properties parameters ()
* Fix header ()
* fix horizon bug in eval framework ()
* Fix imports ()
* Fix issue in get_observation() ()
* fix linting ()
* fix linting update pylint and yapf... ()
* Fix merging of done_info ()
* fix Nonetype and None used in required units causing issues ()
* Fix obs space check for inference ()
* fix pipeline error ()
* fix platforms without agents ()
* fix poetry ()
* fix poetry ()
* fix relaserc ()
* fix rllib_experiment debug checker in wrong spot ()
* fix shared_done_info to properly propagate ()
* Fix sps ()
* fix sps2 ()
* fix the normalization issue for repeated spaces ()
* fix throughput issues for observation space generation ()
* fix trainable class check ()
* Fix type ignore ()
* fix typing and allow obsrve_sensor to handle more than just arrays (not... ()
* Fix unit test naming issue ()
* Fix validator ()
* fix value error from nvidia-smi being on the cpu nodes ()
* Fix versioning problems for packages. Updates to docker to support development environments. ()
* fix yaml error ()
* fix-6dof-platform --- approved by BKH and CL ()
* fix-custom-policy-reset-time ()
* fix-mean-sample ()
* fix-memory-store-initialization ()
* fix-policy-checkpoint-loading ()
* fixed configs to work with new interface ()
* fixed dict -> ordereddict ()
* Fixed episode_id ()
* fixed issue where the default action wasn't handled properly in conjunction with multiple agents ()
* fixed issue with function call arg vs kwarg ()
* fixed issue with moving to ray; and minor issue with episode done_string ()
* fixed issue with state ()
* fixed issues for hierarchical learning ()
* Fixed local mode being ignored for debugging ()
* Fixed precision warning ()
* Fixed unit representation issue ()
* Fixes for the simulator reset dictionary ()
* flatten arrays into new fields ()
* force build of reverting cuda update ()
* force mypy to pass in ci ()
* forgot to add scripted_action test policy ()
* frame rate processing for real time ()
* get_sub_environments() is always empty for inference client ()
* **gitignore:** Added ray_logs to gitignore ()
* **gitignore:** Added vscode directory to gitignore ()
* **gitlab-ci:** add stage ()
* **gitlab-ci:** allow mkdocs to fail ()
* **gitlab-ci:** buildkit release ()
* **gitlab-ci:** Change stages ()
* **gitlab-ci:** change to the pypi ()
* **gitlab-ci:** minor change ()
* **gitlab-ci:** minor change to kickoff pipeline ()
* **gitlab-ci:** minor change to test pipeline ()
* **gitlab-ci:** missed a plugin ()
* **gitlab-ci:** missed busybox ()
* **gitlab-ci:** mkdocs job crashing ()
* **gitlab-ci:** monor change to force pipeline ()
* **gitlab-ci:** put script back to old way ()
* **gitlab-ci:** re-add slashes and update build args ()
* **gitlab-ci:** remove slash ()
* **gitlab-ci:** sigh, args back to old way ()
* **gitlab-ci:** try to fix pages ()
* **gitlab-ci:** turn off mkdocs job ()
* **gitlab-ci:** update cicd settings for both envs ()
* **gitlab-ci:** update job names ()
* **gitlab-ci:** update kaniko executor string ()
* **gitlab-ci:** update kaniko-args and version ()
* **gitlab-ci:** update mkdocs installs ()
* **gitlab-ci:** update mkdocs job ()
* **gitlab-ci:** update other script ()
* **gitlab-ci:** update tagged image ()
* **gitlab-ci:** update tagged jobs for buildkit ()
* **gitlab-ci:** update to new pipeline ()
* **gitlab-ci:** yaml lint fix ()
* Glue extractors ()
* handle case when there's no info dict ()
* hot fix to update epp parameter signature in experiment to match for save and load ()
* hot fix to update epp parameter signature in experiment to match for save and load ()
* hot fix to update epp parameter signature in experiment to match for save and load ()
* Hotfix ray 2 ()
* HTML Plot Visualization ()
* if no class defined default to local ()
* ignore new mypy version errors ()
* Ignore pylint error ()
* Improved TabularParameterProvider validators ()
* Incorporate changes from act3 agents ()
* install curl ()
* interface issue with agent validation test ()
* issue with indexing of box spaces ()
* Limit worker set to healthy workers ()
* lint fix: ()
* linting ()
* linting ()
* linting ()
* linting ()
* Log at bound successes ()
* Logic improvements ()
* made normalization automatic; fixed raw_obs issue ()
* make _add_git_hashes_to_config() and_update_rllib_config() private. ()
* Make dependent parameters a dictionary ()
* Make error checking more precise ()
* Make filepath parsing recursive ()
* Make plugin library fail on partial incompatible match ()
* make release.sh +x and undo revert ()
* March cicd updates ()
* Merge branch '126-evaluator-supports-explore-false' into 'main' (), closes
* Merge branch '135-evaluation-framework-can-output-multiple-visualizations' into 'main' (), closes
* Merge branch '143-acedt-integration-integrate-acedt-utility-to-evaluation-framework-pipeline' into 'main' (), closes
* Merge branch '145-add-obs-space-and-units-to-call-for-done-conditions' into 'main' (), closes
* Merge branch '168-temp-hack' into 'main' ()
* Merge branch '203-parametervalidator-dependent_parameters-typing' into 'main' (), closes
* Merge branch '43-add-debug-flag-to-train_rl' into 'main' (), closes
* Merge branch '47-validators-as-properties' into 'main' (), closes
* Merge branch '48-exclusiveness-as-property' into 'main' (), closes
* Merge branch '49-baseplatformpart-validator' into 'main' (), closes
* Merge branch '67-observesensor-output-units' into 'main' (), closes  
* Merge branch '74_configurable_sanity_check' into 'main' (), closes
* Merge branch '***-163v2' into 'main' ()
* Merge branch 'add_create_training_observations_fallback' into 'main' ()
* Merge branch 'auto_updates' of *****:act3-rl/act3-rl-core into auto_updates ()
* Merge branch 'avoid_double_platform_delete' into 'main' ()
* Merge branch 'beta' of *****:act3-rl/corl into beta ()
* Merge branch 'cleanup/references_sims' into 'main' ()
* Merge branch 'delete-moved-controller-wrappers' into 'main' ()
* Merge branch 'evaluation-param-check' into 'main' ()
* Merge branch 'evaluation-remote-support' into 'main' ()
* Merge branch 'feature/edge_case_considerations' into 'main' ()
* Merge branch 'feature/eval_capture_environment_state' into 'main' ()
* Merge branch 'feature/eval-visualization' into 'main' ()
* Merge branch 'fix/nvidia-smi-resource-error-cpu' into 'main' ()
* Merge branch 'fix/package2' into 'main' ()
* Merge branch 'fix/rel_parameters_position' into 'main' ()
* Merge branch 'fix/setup.py' into 'main' ()
* Merge branch 'fix/update_correct_names' into 'main' ()
* Merge branch 'fix/update_correct_names' into 'main' ()
* Merge branch 'ft_worth' into 'main' ()
* Merge branch 'main' of *****:act3-rl/act3-rl-core into main ()
* Merge branch 'main' of *****:act3-rl/act3-rl-core into main ()
* Merge branch 'make_run_experiment_loc_smaller' into 'main' ()
* Merge branch 'precision-warning' into 'main' ()
* Merge branch 'sdf/remove-underscore-naming-convention' into 'main' ()
* Merge branch 'simulator-validators-as-properties' into 'main' ()
* Merge branch 'test/memory_store' into 'main' ()
* Merge branch 'typeerror-fix' into 'main' ()
* Merge branch 'update/vista_updates' into 'main' ()
* Merge branch 'update/vista_updates' into 'main' ()
* Merge branch 'vista_challenge_problem_1' into 'main' ()
* Merge branch 'vista_challenge_problem_1' into 'main' ()
* Merge branch 'vista_challenge_problem_1' into 'main' ()
* merge in the master to branch ()
* merge main into 620-update ()
* Merge remote-tracking branch 'origin/main' into 156-cumulative-reward-by-source-custom-metric ()
* Merge remote-tracking branch 'origin/main' into 83-autodetect-hpc-system-vs-local-system ()
* Merge remote-tracking branch 'origin/main' into auto_updates ()
* Merge remote-tracking branch 'origin/main' into dca-1v1 ()
* Merge remote-tracking branch 'origin/main' into docking1d ()
* Merge remote-tracking branch 'origin/main' into fix-hierarchical-learning ()
* Merge remote-tracking branch 'origin/main' into poetry_experiments ()
* Method to embed properties ()
* minor change ()
* Minor error handling improvements ()
* modified 'controllers' definition to not use a dict ()
* More validators as properties ()
* Move factory to separate library ()
* Move function out of loop; add exception logging ()
* Move functors to library ()
* move to latest code server 4.0.2 - tested local ()
* move to ray 1.13.0 and make callbacks a plugin system ()
* move to use foreach call ()
* moved obs data to info dict ()
* Moved reset from ScriptedAction policy to CustomPolicy ()
* moved resource var to project var, apply it to all pipeline stages ()
* Moved sensorconfig ()
* mypy ()
* mypy ()
* mypy ()
* mypy changes ()
* nan check applied controls ()
* Noop With Arguments ()
* observation_space() uses create_converted_space() ()
* Parameter provider tests and fixes ()
* **parameter:** Add some typing ()
* **parameter:** Fix issue with test_choices_string() ()
* **parameter:** Force return of native types ()
* **parameter:** Make get_validator() not abstract ()
* **parameter:** Relocated bound_func attribute ()
* **parameter:** Removed get_native() method, now automatically handled by units class ()
* Path and usability fixes for evaluation ()
* Pin mkdocstrings to 0.18.0 ()
* pin packages ()
* Platform utilities ()
* Platform utilities actually used ()
* poetry to use a single gym ()
* poetry updates ()
* post develop ()
* Prohibit EpisodeLengthDone in agent dones ()
* Proper method call ()
* Properly handle partial output units specified ()
* push hot fix for model search ()
* Put plugin path in cartpole configuration ()
* pylint ()
* Pylint ()
* pylint fix ()
* pytlint ()
* Raise instance not class ()
* **ray:** update ray to 1.9 ()
* re-add slashes ()
* relative parameters for the position ()
* remove command used for testing ()
* Remove debugging print statements ()
* Remove default_min, default_max ()
* remove dockerfile poetry installs ()
* remove evil print statement ()
* remove focal ()
* remove focal ()
* remove focal ()
* remove learjet base controllers ()
* remove learjet configs ()
* Remove MainUtil dependency ()
* remove MultiBox ()
* Remove np.bool_ from DoneDict ()
* remove one cartpole until investigation is done ()
* remove pickle5 because ray was whining about it ()
* Remove position hack ()
* remove prop validation ()
* Remove redundant lines in reset and fixes the observer sensor so that you can set names ()
* remove runtime error from eval pipeline checkpoint loading ()
* Remove unused import ()
* Remove unused pydantic models ()
* remove unwanted needs keyword ()
* remove utf encoding in binary ()
* Remove websocat stage ()
* remove words ()
* removed )
* Removed comment ()
* Removed comment ()
* Removed np.array ()
* removed override annotation ()
* Removed pylint disable ()
* Removed type ignore ()
* Removed unneeded inits ()
* Removed unneeded min()/max() ()
* Removed unneeded property ()
* Removed unused min() and max() ()
* **remove:** remove pages changes ()
* Removing _set_all_done() ()
* removing apt mirror for now ()
* Rename environment glue/reward/done creation methods ()
* rename file ()
* rename method ()
* rename policy name to policy id ()
* Rename properties to property_class ()
* Renamed Percent to PartOfWhole, added fraction ()
* Replace constr with Annotated ()
* Replaced list with abc.Sequence ()
* Replaced TestProp with BoxProp ()
* require tf until rllib fixes issues with tf_prob. Also updated lock file ()
* **requirements:** adrpy missing ()
* **requirements:** update imports ()
* Resolve "add agent_platform names to reward base config / initialization args" ()
* Resolve "add git hashes to env_config" ()
* Resolve "Add obs space and units to call for done conditions" ()
* Resolve "Add validation to base agent glues that reference parts exist" ()
* Resolve "Create simulator callbacks" ()
* Resolve "Environment simulator parameter magic strings" ()
* Resolve "Evaluation metric for done percentages" ()
* Resolve "Evaluation uses episode state with agent keys rather than platform keys" ()
* Resolve "Log checkpoint repository hash" ()
* Resolve "Log short episodes to tensorboard" ()
* Resolve "ParameterValidator.dependent_parameters typing" ()
* Resolve "Use actual units rather than configured for OverridableParameterWrapper" ()
* revert agents base version and remove chrome ()
* revert back to tf 2.4.0 given tcn issues ()
* revert back to tf 2.4.1 given tcn issues ()
* revert changes that disabled auto rollout fragments ()
* revert changes to callbacks; now handling this issue in the inference experiment class ()
* revert encoding fail from linting ()
* reverted changes w.r.t. disabling normalization via the policy_config ()
* Reward path ()
* **rewards:** fixed post_process_trajectory reward function signature ()
* Rework types for pylint ()
* Rewrite directory handling to avoid library module ()
* rllib 1d docking ()
* sdf/comms ()
* sdf/fix-custom-policy-batches ()
* sdf/fix-overridable-parameter ()
* sdf/inference2.0 ()
* sdf/remove-underscore-naming-convention ()
* second try at exclusivity bypass ()
* semantic release roll back ()
* **setup.py:** changes for MR ()
* **setup.py:** remove MIT line ()
* **setup.py:** remove package dir ()
* **setup.py:** remove package dir ()
* **setup.py:** trying the old one with a mod ()
* **setup.py:** update for pip installs ()
* **setup.py:** update for pip installs ()
* **setup.py:** update to fix import errors ()
* **setup:** minor errors ()
* Silly bugs ()
* **simulator:** fix the environment not passing worker index and vector index to simulator correctly ()
* slim controller glue config ()
* small bounds tweak ()
* small changes ()
* small updates for code to support HLP - small cleanup i ()
* sources.list changes ()
* sources.list changes ()
* sources.list changes ()
* spelling ()
* start of an export excel setup/experiment ()
* Start on glues and rewards ()
* streamlit-app with changes to episode-artifact ()
* Streamlit: Agent Steps Options ()
* style ()
* Switch to pathlib.Path ()
* syntax error ()
* TabularParameterProvider related typing ()
* temporarily disable pages ()
* test ()
* test ()
* test buildkit ()
* test pass for dict wrapper glue ()
* **test:** add test for 2d boxprop ()
* the following fixes formatting... ()
* The following updates move to the latest docker images and support setup of act3_corl. ()
* traces with units disappear ()
* trigger done for all agents on a deleted platform ()
* trigger release ()
* try again to use agent base 2.2.1 ()
* trying a differetn url ()
* trying to fix semantic release ()
* trying to get a new runner - updates docs ()
* turn of mkdocstring for multi_agent_env ()
* typing and formatting pipeline fails ()
* typos ()
* Units and aero library ()
* **units:** Added strict types and fixed handling of none unit ()
* **units:** Added value validator to automatically convert numpy types ()
* unwanted txt got into requirement.dep file. edit command to fix that ()
* update agent base to new path and update ray to 1.12.1 ()
* update apt cache ()
* update apt-mirror-url var ()
* update apt-mirror-url var ()
* update apt-mirror-url var ()
* update branch ()
* update cicd to use act3 jobs and make release image ()
* update cicd, add semantic updates ()
* Update comment ()
* Update controllers and platform parts also ()
* update doc strings ()
* update docker file to use poetry ()
* Update Dockerfile ()
* Update Dockerfile ()
* Update Dockerfile ()
* Update Dockerfile ()
* Update Dockerfile ()
* Update embed_propreties to new variable names ()
* update external deps ()
* update external deps ()
* update external deps ()
* update external deps ()
* update files and scripts for poetry ()
* update for task needed for flight tests ()
* Update gitlab pipeline version ()
* update image paths, changes for 620 ()
* update install poetry command and fix poetry export command ()
* update linting... ()
* Update multi_agent_env.py ()
* Update multi_agent_env.py - removed useless return ()
* Update observation_units() ()
* Update observation_units() ()
* update oci_registry for buildkit, make changes for docker-compose ()
* update permissions for hpc image in rl- ()
* update pip_index_url ()
* Update plugin library dependency ()
* update poetry python version ()
* update ray ()
* update ray to 1.12 ()
* Update setup.py ()
* Update shape typing ()
* update sources.list ()
* update sources.list ()
* update sources.list ()
* update syntax and dockerfile ()
* update tag ()
* update to add autodetect of local vs hpc system ()
* update to add css ()
* update to allow reset of agent config as part of reset of simulator ()
* update to auto setup ()
* update to correct circular dep ()
* update to correct mkdocs?? ()
* update to correct names for targets in .gitlab-ci.yml ()
* update to fix scipy export issue ()
* update to get box prop working ()
* Update train_rl help messages ()
* update types for mypy ()
* update typing ()
* Update warmup ()
* Update/vista updates ()
* updated and ensured that it also works with the cartpole env ()
* Updated build_episode_parameter_provider_parameters() ()
* updated custom policy to use info dict ()
* updated epp access to be from the config ()
* updated key from glues->dones ()
* Updated ObserveSensor unit tests ()
* Updated OpenAIGymSimulator init ()
* Updated output_units validation ()
* Updated typing ()
* Updated unit test ()
* Updated unit test ()
* Updated units and spaces ()
* updates ()
* updates adds in the ability to restore based on the new setup in RAY.... ()
* updates for APRS ()
* updates for APRS ()
* updates for auto code reporting ()
* updates for auto test ()
* updates for challenge problem )
* updates for challenge problem on LLA ()
* updates for code base ()
* updates for code base... ()
* updates for conflicts ()
* updates for *** ()
* updates for e4xternal deps ()
* updates for external deps ()
* updates for external deps ()
* updates for HL example ()
* updates for lint ()
* updates for lint and just added docs updates ()
* updates for linting ()
* updates for linting ()
* updates for linting ()
* updates for memstore and HL ()
* updates for memstore and HL ()
* updates for merge ()
* updates for numpy bool ()
* updates for orginizatio, linting and testing ()
* updates for population style training with PPO ()
* updates for the code server and test for *** push ()
* updates for the league --- TEMP until we redesign restart ()
* Updates for the memory store, valid states, and HL ()
* updates for the memstore ()
* Updates for VISTA model ()
* updates linting ()
* updates required for  ()
* updates so that league aers go to the same dir every time --- skip the date and pbs keys ()
* updates to (1) /tmp/data vs /opt/data supporting *** exploration and (2)... ()
* updates to add in clutter ()
* updates to add in container building ()
* updates to add in missing items and set path ()
* updates to add in profile examples ()
* updates to add in vscode builds ()
* updates to add obs to the apply at agent level ()
* updates to allow push to *** - temp fix --- also fix the linting ()
* updates to controller to report invalid state correctly for non len 1 arrays ()
* updates to docker ()
* updates to docker ()
* updates to docker ()
* updates to get code working ()
* updates to get docs ()
* updates to how we create the topological search ()
* updates to linting ()
* updates to make pluggable class for autodetect based on program needs ()
* updates to make pluggable class for autodetect based on program needs ()
* updates to make sure the setup.py and the requirements.txt are synced ()
* updates to make the pylint items identifiable and to add in profiling ()
* updates to remove 2 containers ()
* updates to remove files that are not in main ()
* updates to remove patches since not needed still ()
* updates to tensorflow 2.7.0 cuda 11.1 torch 1.8 ()
* updates to the build number version ()
* updates to the code base ()
* updates to the code base for pdf ()
* updates to the code base to allow skipping setup of the simulator on reinit. ()
* use apt install to install pip ()
* use dst name instead of src name during unit resolution ()
* use glue name since target value can vary with ADR ()
* use glue name since target value can vary with ADR ()
* Use new buffer size variable ()
* use policy config ()
* use pytest ()
* use sim_time from state, not sim ()
* using new semantic release image ()
* validate control against gym space ()
* Validate units ()
* Validators as properties ()
* Value function analysis ()
* Vista challenge problem 1 ()
* Vista challenge problem 1 ()
* Vista challenge problem 1 ()
* **vscode:** fix debugging in vscode ()
* **warning:** Fix precision warning ()
* **warning:** Fix precision warning ()
* **warning:** Fix precision warning ()

* Merge branch 'commander_agent_class' into 'main' ()

### Features

* (wip) added scripted_action policy ()
* add container stages for release ()
* Add controller wrappers ()
* add done status callback ()
* Add episode reward ()
* Add EpisodeLengthDone, SensorBoundsCheckDone ()
* add exclusiveness checking to controllers ()
* add me parts test ()
* add no op agents and controllers ()
* add non trainable base ()
* add non-trained policy support for rllib experiments ()
* add platform valdiation and remove required internal platform ()
* add plugin lib ()
* add policy mapping configuration ()
* add pong simulator ()
* Add support for evaluation ()
* add support for loading from checkpoints into an agent policy ()
* Add TabularParameterProvider ()
* Add TAS/CAS units ()
* add TestCaseManager to evaluator to allow extensible configuration options ()
* add tests ()
* add visualizations for the pong simulator ()
* added a do nothing callback ()
* added convert_to method directly onto the ValueWithUnits validator class ()
* added in configuration to enable per-agent policy configurations ()
* added init files ()
* Added observation filtering ()
* added part validity concept as well as getter/setter functions for validity ()
* Added RllibExperiment get_callbacks method ()
* Adds plumbing for KIAS sensor ()
* adds repeated observe sensor and supports openaigym repeated ()
* Agent episode parameter providers ()
* Automatic Domain Randomization ()
* avoid adding policies for agents not accessed by rllib ()
* Better reference handling ()
* caught spelling mistake in imports ()
* change act3_rl_core to corl ()
* changed EpisodeLength to EpisodeLength_Steps , episode length in steps not realtime ()
* configs for docking 1d eval ()
* configs for evaluation and 1d docking env ()
* Connect dones to new functor argument ()
* define spaces as props held by parts ()
* **dones:** added shared dones, fixed aer strings ()
* Enable EPP for simulation reset ()
* Enabled parts to be both a sensor and controller simultaneously; added part - MemoryStore, which stors data locally (i.e. not in the sim) that can be added to the action or obs space of any agent on the platform, thus providing a mechanism to communicate a single value ()
* Encapsulate agent spaces ()
* enforce an agent/platform naming convention ()
* Environment episode parameter providers ()
* Environment parameters passed to agent ()
* Eval framework docs ()
* evaluation framework, switched changed appropriate refs to corl ()
* fix package ()
* fixed 1d docking serialize function ()
* foot / second^2 added as acceleration unit ()
* functor and base glue/reward/done ()
* get rid of unneeded typing.Union ()
* **glue:** added glue multi wrapper base class ()
* handling for checkpoint types beside rllib checkpoint, h5 and heuristic ()
* Initial copy from *** act3 core ()
* initial pass at dependent random variables ()
* make curl stage external container ()
* Merge branch '17-update-ray-to-version-1.7' into 'main' (), closes
* Merge branch '7-agent-config-variable-store' into 'main' (), closes
* Merge branch 'evaluation-support' into 'main' ()
* Merge branch 'fix-container-package' into 'main' ()
* Merge branch 'fix/package2' into 'main' ()
* Merge branch 'move_episode_reward_to_core' into 'main' ()
* Merge branch 'tas_cas_unit' into 'main' ()
* move obs creation for future type specification feature ()
* move to ME parts to platform ()
* MultiMeasurementOperation class ()
* new arrangement for 1d docking agent ()
* Operable platforms ()
* **parameter:** Switch to using the core Parameter class ()
* path updates and fixes ()
* **plugin library:** update how plugin library works/extracts ()
* plumbing for aer file writing ()
* Pong Sac Discrete Actions ()
* Pong SAC Flatten Normalized Actions ()
* precommit pipeline fix ()
* provide obs_units to draw/annot functor ()
* provided a config update that does nothing ()
* Pydantic checking of ACT3MultiAgentEnv ()
* Pydantic validation for environments ()
* reference fix ()
* reference store deeper config use, added lots of data for glue extractors to use ()
* Registry ()
* relaxed some abstract functions in base glue, added ObservePartValidity ()
* Remote support for episode parameter providers ()
* remove imports from init files ()
* remove non trinables ()
* remove unneeded metrics ()
* removed unneeded metric generator definitions ()
* removed winning team metric ()
* Resolve "remove simulator wrapper" ()
* **rewards:** implemented post_process_trajectory call ()
* rewrote obs extractor to simplify interface ()
* running successfully with multiple agents per platform ()
* Sensor bounds done and delayed unit resolution ()
* serialization for specific platforms 1d docking ()
* setup a default for the eval_config_update and changed it appropriately in 1d docking config file ()
* **simulator:** gym simulator accepts env kwargs ()
* space definition and units table ()
* streamlit unit conversion ()
* streamlit visualization ()
* switched around simulator reset parameters to exist in docking1d_env not in docking1d_agent config ()
* **test:** Added units unit test ()
* track episode length by number of steps not seconds ()
* trajectory animation ()
* Unit conversion and storage parameters in Functor ()
* update corl to ray 2 ()
* update interface to simulator.mark_episode_done to allow more fine tuned simulator processing of info ()
* Update lcm calculation for integer agent periods ()
* update paths and made appropriate fixes ()
* update paths and type hints for evaluate_from_config pathways ()
* update paths for cli_generate_metrics pathway ()
* update to EpisodeLength_Steps metric in config ()
* update to python 3.10 ()
* update to the 1d docking eval config ()
* update to using ValueWithUnits in Validator and for building state ()
* updated docs and moved places ()
* updated launch commands and metrics config for 1d Docking ()
* updated visualize and print pathways and fixed mypy ()
* Utilize validators for done conditions ()
* working docking 1d test cases config ()

### BREAKING CHANGES

* CoRL 2.0

See merge request act3-rl/corl!317

## 2.13.8

### Bug Fixes

* Streamlit: Agent Steps Options ()

## 2.13.7

### Bug Fixes

* Make filepath parsing recursive ()

## 2.13.6

### Bug Fixes

* configurable state pickle writing on obs fail ()

## 2.13.5

### Bug Fixes

* streamlit-app with changes to episode-artifact ()

## 2.13.4

### Bug Fixes

* Resolve "Log short episodes to tensorboard" ()

## 2.13.3

### Bug Fixes

* Value function analysis ()

## 2.13.2

### Bug Fixes

* Remove redundant lines in reset and fixes the observer sensor so that you can set names ()

## 2.13.1

### Bug Fixes

* Eval framework - Namespace TypeError fix for visualization launcher ()
* Merge branch 'typeerror-fix' into 'main' ()

# 2.13.0

### Bug Fixes

* Fix annotation state alignment ()
* update for task needed for flight tests ()

### Features

* provide obs_units to draw/annot functor ()

## 2.12.1

### Bug Fixes

* adds step time in seconds and steps to episode metadata ()

# 2.12.0

### Features

* trajectory animation ()

# 2.11.0

### Features

* streamlit unit conversion ()

## 2.10.8

### Bug Fixes

* Error handling in epp validator ()

## 2.10.7

### Bug Fixes

* Resolve "Use actual units rather than configured for OverridableParameterWrapper" ()

## 2.10.6

### Bug Fixes

* sdf/fix-overridable-parameter ()

## 2.10.5

### Bug Fixes

* traces with units disappear ()

## 2.10.4

### Bug Fixes

* Add Units and Space Definition to Default Callback ()

## 2.10.3

### Bug Fixes

* Make dependent parameters a dictionary ()

## 2.10.2

### Bug Fixes

* fix horizon bug in eval framework ()

## 2.10.1

### Bug Fixes

* Merge branch '203-parametervalidator-dependent_parameters-typing' into 'main' (), closes
* Resolve "Environment simulator parameter magic strings" ()
* Resolve "ParameterValidator.dependent_parameters typing" ()

# 2.10.0

### Features

* space definition and units table ()

## 2.9.1

### Bug Fixes

* flatten arrays into new fields ()

# 2.9.0

### Bug Fixes

* Update gitlab pipeline version ()

### Features

* streamlit visualization ()

## 2.8.18

### Bug Fixes

* revert changes that disabled auto rollout fragments ()

## 2.8.17

### Bug Fixes

* change order of no-debug flag to fix default settings ()

## 2.8.16

### Bug Fixes

* updates to how we create the topological search ()

## 2.8.15

### Bug Fixes

* updates to (1) /tmp/data vs /opt/data supporting *** exploration and (2)... ()

## 2.8.14

### Bug Fixes

* sdf/comms ()

## 2.8.13

### Bug Fixes

* Resolve "Create simulator callbacks" ()

## 2.8.12

### Bug Fixes

* start of an export excel setup/experiment ()
* updates to the code base ()

## 2.8.11

### Bug Fixes

* remove words ()
* updates for code base ()

## 2.8.10

### Bug Fixes

* do not return empty obs for platforms without any glues ()

## 2.8.9

### Bug Fixes

* fix sps2 ()

## 2.8.8

### Bug Fixes

* Fix sps ()

## 2.8.7

### Bug Fixes

* enable different parts and glue on each platform in multiplatform agent ()

## 2.8.6

### Bug Fixes

* fix throughput issues for observation space generation ()
* March cicd updates ()
* remove evil print statement ()
* update to fix scipy export issue ()
* Update warmup ()
* updates adds in the ability to restore based on the new setup in RAY.... ()

## 2.8.5

### Bug Fixes

* Ability to evaluate array action space ()

## 2.8.4

### Bug Fixes

* fix typing and allow obsrve_sensor to handle more than just arrays (not... ()

## 2.8.3

### Bug Fixes

* sdf/fix-custom-policy-batches ()

## 2.8.2

### Bug Fixes

* Limit worker set to healthy workers ()
* Proper method call ()

## 2.8.1

### Bug Fixes

* Make error checking more precise ()

# 2.8.0

### Features

* Update lcm calculation for integer agent periods ()

## 2.7.10

### Bug Fixes

* The following updates move to the latest docker images and support setup of act3_corl. ()

## 2.7.10-beta.5

### Bug Fixes

* Framer rate step arg ()

## 2.7.10-beta.4

### Bug Fixes

* updates to match github ()
* update readme ()
* updates ()
* updates for cleanup ()
* updates for cleanup ()
* updates for git ()
* updates for git ()

## 2.7.10-beta.3

### Bug Fixes

* updates for the develop target ()

## 2.7.10-beta.2

### Bug Fixes

* Merge remote-tracking branch 'origin/main' into beta ()

## 2.7.10-beta.1

### Bug Fixes

* clean up scripts ()
* clean up scripts ()
* edit releaserc ()
* increase cpu on unit test ()
* lower kube resource ()
* Merge remote-tracking branch 'origin/main' into new_container_setup ()
* move to the latest version ()
* remove leauge from this ()
* revert back the req file ()
* update ci file and add coverage-badge ()
* update lock ()
* update poetry.lock and test release ()
* updates ()
* updates ()
* updates ()
* updates ()
* updates ()
* updates ()
* updates ()
* updates docs ()
* updates for ci ()
* updates for ci ()
* updates for corl new images ()
* updates for corl new images ()
* updates for corl new images ()
* updates for docs ()
* updates to new containers ()

## 2.7.9

### Bug Fixes

* Eval episode artifact ()

## 2.7.8

### Bug Fixes

* Eval Fixes ()

## 2.7.7

### Bug Fixes

* remove runtime error from eval pipeline checkpoint loading ()

## 2.7.6

### Bug Fixes

* fix-policy-checkpoint-loading ()

## 2.7.5

### Bug Fixes

* bugfix launch pipeline.py ()

## 2.7.4

### Bug Fixes

* fix frozen weight loading not working, and allow target value diff to ignore invalid parts ()

## 2.7.3

### Bug Fixes

* eval default callbacks fix ()
* fix shared_done_info to properly propagate ()
* update cicd, add semantic updates ()

## 2.7.2

### Bug Fixes

* Resolve "Add validation to base agent glues that reference parts exist" ()

## 2.7.1

### Bug Fixes

* slim controller glue config ()

# 2.7.0

### Features

* update to python 3.10 ()

## 2.6.4

### Bug Fixes

* fix platforms without agents ()

## 2.6.3

### Bug Fixes

* fix commander pong eval and bugs in iterate_test_cases ()

## 2.6.2

### Bug Fixes

* eval hotfix ()

## 2.6.1

### Bug Fixes

* APRS 2 Release ()

# 2.6.0

### Features

* add support for loading from checkpoints into an agent policy ()

## 2.5.2

### Bug Fixes

* sdf/inference2.0 ()

## 2.5.1

### Bug Fixes

* cleanup-eval-initialization ()

# 2.5.0

### Features

* reference store deeper config use, added lots of data for glue extractors to use ()

## 2.4.1

### Bug Fixes

* Config cleanup for public release ()

# 2.4.0

### Features

* add visualizations for the pong simulator ()

# 2.3.0

### Features

* add TestCaseManager to evaluator to allow extensible configuration options ()

# 2.2.0

### Features

* Operable platforms ()

## 2.1.2

### Bug Fixes

* All platforms dones ()

## 2.1.1

### Bug Fixes

* Resolve "add agent_platform names to reward base config / initialization args" ()

# 2.1.0

### Features

* Resolve "remove simulator wrapper" ()

# 2.0.0

* Merge branch 'commander_agent_class' into 'main' ()

### BREAKING CHANGES

* CoRL 2.0

See merge request act3-rl/corl!317

## 1.60.1

### Bug Fixes

* require tf until rllib fixes issues with tf_prob. Also updated lock file ()

# 1.60.0

### Features

* Pong Sac Discrete Actions ()

# 1.59.0

### Bug Fixes

* Hotfix ray 2 ()
* poetry to use a single gym ()
* turn of mkdocstring for multi_agent_env ()

### Features

* move obs creation for future type specification feature ()
* update corl to ray 2 ()

# 1.58.0

### Features

* Pong SAC Flatten Normalized Actions ()

# 1.57.0

### Features

* Encapsulate agent spaces ()

# 1.56.0

### Features

* add pong simulator ()

# 1.55.0

### Features

* initial pass at dependent random variables ()

## 1.54.8

### Bug Fixes

* Resolve "Evaluation metric for done percentages" ()

## 1.54.7

### Bug Fixes

* attempt to get torch version down to 1.20.0 ()

## 1.54.6

### Bug Fixes

* add glue obs clipping functionality and add AT_LEAST DoneStatusCode tensorboard output ()

## 1.54.5

### Bug Fixes

* fix-custom-policy-reset-time ()
* fix-mean-sample ()

## 1.54.4

### Bug Fixes

* fix the normalization issue for repeated spaces ()

## 1.54.3

### Bug Fixes

* add flag to disable auto rllib config setup ()
* deprecate agent_name == platform_name ()
* remove dockerfile poetry installs ()

## 1.54.2

### Bug Fixes

* add gputil ()

## 1.54.1

### Bug Fixes

* update files and scripts for poetry ()

# 1.54.0

### Features

* Eval framework docs ()

## 1.53.4

### Bug Fixes

* rename policy name to policy id ()

## 1.53.3

### Bug Fixes

* remove utf encoding in binary ()

## 1.53.2

### Bug Fixes

* revert encoding fail from linting ()

## 1.53.1

### Bug Fixes

* fix-memory-store-initialization ()

# 1.53.0

### Features

* add policy mapping configuration ()

## 1.52.9

### Bug Fixes

* fix-6dof-platform --- approved by BKH and CL ()

## 1.52.8

### Bug Fixes

* fix linting update pylint and yapf... ()

## 1.52.7

### Bug Fixes

* custom policy interface changes to support inference ()

## 1.52.6

### Bug Fixes

* add seed argument to create space ()

## 1.52.5

### Bug Fixes

* correct the eval pipeline to use correct paths ()

## 1.52.4

### Bug Fixes

* 163v2 ()
* add debug mode ()
* add flag on the wrong command ()
* add index url ()
* add missing install command ()
* add path to install ()
* add poetry pyproject.toml and poetry.lock ()
* add requirement.lock file to semantic assets ()
* add trial_str_functor plugin and fix verbosity argument to train_rl which was broken ()
* Allow Evaluator to specify multiple rllib config updates ()
* attempt 2 at docker + poetry ()
* edit install poetry command ()
* fix pipeline error ()
* fix poetry ()
* fix poetry ()
* fix yaml error ()
* install curl ()
* Merge branch '-163v2' into 'main' ()
* Merge branch 'beta' of *****:act3-rl/corl into beta ()
* Merge remote-tracking branch 'origin/main' into poetry_experiments ()
* moved resource var to project var, apply it to all pipeline stages ()
* poetry updates ()
* remove command used for testing ()
* remove unwanted needs keyword ()
* test ()
* test ()
* trigger release ()
* trying to fix semantic release ()
* unwanted txt got into requirement.dep file. edit command to fix that ()
* update docker file to use poetry ()
* update install poetry command and fix poetry export command ()
* update poetry python version ()
* updates for merge ()
* updates to add in clutter ()
* updates to add in missing items and set path ()
* use apt install to install pip ()
* using new semantic release image ()

## 1.52.3

### Bug Fixes

* Merge branch '168-temp-hack' into 'main' ()
* Resolve "Evaluation uses episode state with agent keys rather than platform keys" ()

## 1.52.2

### Bug Fixes

* Evaluation framework can output multiple visualizations ()
* Merge branch '135-evaluation-framework-can-output-multiple-visualizations' into 'main' (), closes

## 1.52.1

### Bug Fixes

* Fix obs space check for inference ()

# 1.52.0

### Features

* handling for checkpoint types beside rllib checkpoint, h5 and heuristic ()

## 1.51.1

### Bug Fixes

* Merge branch 'ft_worth' into 'main' ()
* Noop With Arguments ()

# 1.51.0

### Features

* add platform valdiation and remove required internal platform ()

## 1.50.7

### Bug Fixes

* fixed issue with state ()
* frame rate processing for real time ()
* get_sub_environments() is always empty for inference client ()
* handle case when there's no info dict ()
* moved obs data to info dict ()
* revert changes to callbacks; now handling this issue in the inference experiment class ()
* Update multi_agent_env.py ()
* Update multi_agent_env.py - removed useless return ()
* updated custom policy to use info dict ()
* use sim_time from state, not sim ()

## 1.50.6

### Bug Fixes

* Adding environment to extract items from environment state and manipulates accumulate non-terminal metric ()
* Merge branch 'feature/eval_capture_environment_state' into 'main' ()

## 1.50.5

### Bug Fixes

* Acedt integration with evaluation framework ()
* Merge branch '143-acedt-integration-integrate-acedt-utility-to-evaluation-framework-pipeline' into 'main' (), closes

## 1.50.4

### Bug Fixes

* HTML Plot Visualization ()
* Merge branch 'feature/eval-visualization' into 'main' ()

## 1.50.3

### Bug Fixes

* **callbacks:** cumulative reward per reward source added to custom metrics ()
* Merge remote-tracking branch 'origin/main' into 156-cumulative-reward-by-source-custom-metric ()
* Remove websocat stage ()

## 1.50.2

### Bug Fixes

* adds fallback code to create_training_observations so that agent's that... ()
* Merge branch 'add_create_training_observations_fallback' into 'main' ()

## 1.50.1

### Bug Fixes

* add trial_str_functor plugin and fix verbosity argument to train_rl which was broken ()

# 1.50.0

### Features

* update interface to simulator.mark_episode_done to allow more fine tuned simulator processing of info ()

## 1.49.12

### Bug Fixes

* Edge case consideration ()
* Merge branch 'feature/edge_case_considerations' into 'main' ()

## 1.49.11

### Bug Fixes

* Avoid double platform delete ()
* Merge branch 'avoid_double_platform_delete' into 'main' ()

## 1.49.10

### Bug Fixes

* bump the code-server version and add tags for the GPUs (non specific to nvidia ()

## 1.49.9

### Bug Fixes

* fix relaserc ()
* make release.sh +x and undo revert ()
* semantic release roll back ()
* Update train_rl help messages ()

## 1.49.7

### Bug Fixes

* Merge branch '145-add-obs-space-and-units-to-call-for-done-conditions' into 'main' (), closes
* Resolve "Add obs space and units to call for done conditions" ()

## 1.49.6

### Bug Fixes

* Merge branch 'vista_challenge_problem_1' into 'main' ()
* Vista challenge problem 1 ()

## 1.49.5

### Bug Fixes

* Ensure noop controller uses proper API ()

## 1.49.4

### Bug Fixes

* Merge branch 'vista_challenge_problem_1' into 'main' ()
* Vista challenge problem 1 ()

## 1.49.3

### Bug Fixes

* Merge branch 'vista_challenge_problem_1' into 'main' ()
* Vista challenge problem 1 ()

## 1.49.2

### Bug Fixes

* corrected the implementation of StatusCode and updated 1d docking config ()

## 1.49.1

### Bug Fixes

* updates for challenge problem )

# 1.49.0

### Bug Fixes

* **controller:** continuous gym cntrllers work now ()

### Features

* **simulator:** gym simulator accepts env kwargs ()

## 1.48.11

### Bug Fixes

* Merge branch 'update/vista_updates' into 'main' ()
* Update/vista updates ()

## 1.48.10

### Bug Fixes

* Merge branch 'update/vista_updates' into 'main' ()
* Updates for VISTA model ()

## 1.48.9

### Bug Fixes

* Minor error handling improvements ()

## 1.48.8

### Bug Fixes

* second try at exclusivity bypass ()

## 1.48.7

### Bug Fixes

* **Dockerfile:** update DOCKER_OCI_REG ()
* **gitlab-ci:** mkdocs job crashing ()
* **gitlab-ci:** try to fix pages ()
* **gitlab-ci:** update tagged jobs for buildkit ()
* **remove:** remove pages changes ()

## 1.48.6

### Bug Fixes

* too many lines of code in run_experiment() ()
* **Dockerfile:** update registry arg ()
* **gitlab-ci:** add stage ()
* **gitlab-ci:** buildkit release ()
* **gitlab-ci:** minor change ()
* **gitlab-ci:** minor change to kickoff pipeline ()
* **gitlab-ci:** minor change to test pipeline ()
* **gitlab-ci:** missed busybox ()
* **gitlab-ci:** monor change to force pipeline ()
* **gitlab-ci:** re-add slashes and update build args ()
* **gitlab-ci:** remove slash ()
* **gitlab-ci:** turn off mkdocs job ()
* **gitlab-ci:** update job names ()
* **gitlab-ci:** update mkdocs job ()
* **gitlab-ci:** update to new pipeline ()
* **gitlab-ci:** yaml lint fix ()
* make _add_git_hashes_to_config() and_update_rllib_config() private. ()
* Merge branch 'make_run_experiment_loc_smaller' into 'main' ()
* re-add slashes ()
* Resolve "add git hashes to env_config" ()
* syntax error ()
* temporarily disable pages ()
* test buildkit ()
* trigger done for all agents on a deleted platform ()
* update oci_registry for buildkit, make changes for docker-compose ()

## 1.48.5

### Bug Fixes

* change to nvidia-pytorch ()

## 1.48.4

### Bug Fixes

* added disable_exclusivity_check call to base6dofplatform ()

## 1.48.3

### Bug Fixes

* bugfix to disable_exclusivity ()

## 1.48.2

### Bug Fixes

* Merge branch 'fix/rel_parameters_position' into 'main' ()
* relative parameters for the position ()

## 1.48.1

### Bug Fixes

* move to ray 1.13.0 and make callbacks a plugin system ()
* remove pickle5 because ray was whining about it ()

# 1.48.0

### Features

* added a do nothing callback ()
* changed EpisodeLength to EpisodeLength_Steps , episode length in steps not realtime ()
* configs for docking 1d eval ()
* configs for evaluation and 1d docking env ()
* fixed 1d docking serialize function ()
* get rid of unneeded typing.Union ()
* new arrangement for 1d docking agent ()
* provided a config update that does nothing ()
* setup a default for the eval_config_update and changed it appropriately in 1d docking config file ()
* switched around simulator reset parameters to exist in docking1d_env not in docking1d_agent config ()
* track episode length by number of steps not seconds ()
* update to EpisodeLength_Steps metric in config ()
* update to the 1d docking eval config ()
* update to using ValueWithUnits in Validator and for building state ()
* updated launch commands and metrics config for 1d Docking ()
* working docking 1d test cases config ()

## 1.47.2

### Bug Fixes

* Evaluate supports "explore: False" ()
* Merge branch '126-evaluator-supports-explore-false' into 'main' (), closes

## 1.47.1

### Bug Fixes

* Path and usability fixes for evaluation ()
* Pylint ()
* Switch to pathlib.Path ()

# 1.47.0

### Bug Fixes

* change default value for skip_win_lose_sanity_check ()

### Features

* added init files ()
* caught spelling mistake in imports ()
* path updates and fixes ()
* precommit pipeline fix ()
* remove imports from init files ()
* update paths and made appropriate fixes ()
* update paths and type hints for evaluate_from_config pathways ()
* update paths for cli_generate_metrics pathway ()
* updated visualize and print pathways and fixed mypy ()

## 1.46.4

### Bug Fixes

* added obs_relative_controller_dict ()
* fix FunctorDictWrapper mangling configs of regular wrapper glues ()

## 1.46.3

### Bug Fixes

* use dst name instead of src name during unit resolution ()

## 1.46.2

### Bug Fixes

* Assert on flattened sampled_control ()
* Change agent removed action to random sample ()

## 1.46.1

### Bug Fixes

* fix bug in obs_relative_delta_controller when dealing with multiple actions in a space ()

# 1.46.0

### Bug Fixes

* small bounds tweak ()
* updates to linting ()

### Features

* add done status callback ()

## 1.45.1

### Bug Fixes

* fixed issues for hierarchical learning ()
* Merge remote-tracking branch 'origin/main' into fix-hierarchical-learning ()

# 1.45.0

### Features

* evaluation framework, switched changed appropriate refs to corl ()
* reference fix ()
* remove unneeded metrics ()
* removed unneeded metric generator definitions ()
* removed winning team metric ()
* serialization for specific platforms 1d docking ()
* updated docs and moved places ()

## 1.44.3

### Bug Fixes

* added ability for EpisodeDoneReward to consolidate done conditions and added optional sanity check ()
* fix FunctorDictWrapper validator from crashing on non dict inputs ()

## 1.44.2

### Bug Fixes

* force build of reverting cuda update ()

## 1.44.1

### Bug Fixes

* update agent base to new path and update ray to 1.12.1 ()

# 1.44.0

### Bug Fixes

* Add mkdocstrings)
* Enable part references, functor children ()
* Ensure functor wrappers manage parameters correctly ()
* Pin mkdocstrings to 0.18.0 ()

### Features

* foot / second^2 added as acceleration unit ()

## 1.43.8

### Bug Fixes

* Add output_units validator ()
* Add unit test for ObserveSensorRepeated ()
* Combine output unit validation ()
* Consolidate test sensors ()
* Consolidate test sensors ()
* Fix issue in get_observation() ()
* Fix type ignore ()
* Fix unit test naming issue ()
* Logic improvements ()
* Properly handle partial output units specified ()
* Removed unused min() and max() ()
* Update observation_units() ()
* Update observation_units() ()
* Validate units ()

## 1.43.7

### Bug Fixes

* updates for challenge problem on LLA ()
* updates for linting ()

## 1.43.6

### Bug Fixes

* Allow non-agent platforms ()

## 1.43.5

### Bug Fixes

* post develop ()

## 1.43.4

### Bug Fixes

* change memray to only install on linux ()

## 1.43.3

### Bug Fixes

* Remove np.bool_ from DoneDict ()

## 1.43.2

### Bug Fixes

* Change velocity ned property to mpstas ()

## 1.43.1

### Bug Fixes

* Merge branch 'test/memory_store' into 'main' ()
* Updates for the memory store, valid states, and HL ()

# 1.43.0

### Features

* added part validity concept as well as getter/setter functions for validity ()
* relaxed some abstract functions in base glue, added ObservePartValidity ()

## 1.42.3

### Bug Fixes

* Move function out of loop; add exception logging ()

## 1.42.2

### Bug Fixes

* **gitlab-ci:** update tagged image ()
* update cicd to use act3 jobs and make release image ()

## 1.42.1

### Bug Fixes

* updates for HL example ()
* updates for linting ()

# 1.42.0

### Bug Fixes

* change paths and name to corl ()
* update image paths, changes for 620 ()

### Features

* change act3_rl_core to corl ()

## 1.41.4

### Bug Fixes

* update to add css ()

## 1.41.3

### Bug Fixes

* updates for memstore and HL ()
* updates for memstore and HL ()
* updates for the memstore ()

## 1.41.2

### Bug Fixes

* allow rllib experiment to allow non trainable only runs ()

## 1.41.1

### Bug Fixes

* Merge remote-tracking branch 'origin/main' into auto_updates ()
* rllib 1d docking ()
* updates ()
* updates for APRS ()
* updates for APRS ()
* updates for auto test ()
* updates for conflicts ()
* updates to docker ()
* updates to docker ()
* updates to get code working ()
* updates to remove files that are not in main ()

# 1.41.0

### Bug Fixes

* **Dockerfile:** missed a couple changes ()
* **Dockerfile:** remove ray custom install ()
* **Dockerfile:** replace chmod ()
* **DOckerfile:** update versions on external deps ()
* **Dockerfile:** updates for 620 ()
* fixed issue with moving to ray; and minor issue with episode done_string ()
* **gitlab-ci:** put script back to old way ()
* **gitlab-ci:** sigh, args back to old way ()
* **gitlab-ci:** update cicd settings for both envs ()
* **gitlab-ci:** update kaniko executor string ()
* **gitlab-ci:** update kaniko-args and version ()
* **gitlab-ci:** update other script ()
* linting ()
* Merge branch 'auto_updates' of github.com/act3-ace:act3-rl/corl into auto_updates ()
* Merge branch 'main' of github.com/act3-ace:act3-rl/corl into main ()
* merge main into 620-update ()
* updates for population style training with PPO ()
* updates to docker ()
* updates to the code base for pdf ()

### Features

* remove non trinables ()

## 1.40.2

### Bug Fixes

* Fixed local mode being ignored for debugging ()
* update to auto setup ()
* updates for auto code reporting ()
* updates linting ()

## 1.40.1

### Bug Fixes

* fix rllib_experiment debug checker in wrong spot ()
* push hot fix for model search ()
* remove one cartpole until investigation is done ()
* update ray to 1.12 ()
* updates for code base... ()
* updates for linting ()
* updates to remove patches since not needed still ()

# 1.40.0

### Features

* Enabled parts to be both a sensor and controller simultaneously; added part - MemoryStore, which stors data locally (i.e. not in the sim) that can be added to the action or obs space of any agent on the platform, thus providing a mechanism to communicate a single value ()

## 1.39.3

### Bug Fixes

* updates for numpy bool ()

## 1.39.2

### Bug Fixes

* Merge branch '74_configurable_sanity_check' into 'main' (), closes

## 1.39.1

### Bug Fixes

* Fix merging of done_info ()

# 1.39.0

### Bug Fixes

* changed return type to float ()

### Features

* added convert_to method directly onto the ValueWithUnits validator class ()

## 1.38.4

### Bug Fixes

* auto rllib ()
* auto rllib ()
* if no class defined default to local ()
* Merge remote-tracking branch 'origin/main' into 83-autodetect-hpc-system-vs-local-system ()
* update to add autodetect of local vs hpc system ()
* update to correct circular dep ()
* updates to make pluggable class for autodetect based on program needs ()
* updates to make pluggable class for autodetect based on program needs ()

## 1.38.3

### Bug Fixes

* Added check for properties key ()
* Changed flatten of keys ()
* Fix handling of platform properties parameters ()
* Fixes for the simulator reset dictionary ()
* Updated build_episode_parameter_provider_parameters() ()
* Updated typing ()

## 1.38.2

### Bug Fixes

* Add embedded properties fields ()
* Add observe sensor unit tests ()
* add partofwhole units ()
* Added int to assert ()
* Moved sensorconfig ()
* Replaced TestProp with BoxProp ()
* Update comment ()
* Updated ObserveSensor unit tests ()

## 1.38.1

### Bug Fixes

* A number of changes related to output units ()
* Add 2D support to observation_units() ()
* Add BenchmarkExperiment ()
* Add force units ()
* Add int ()
* Add int ()
* Add int to assert ()
* Add pyinstrument for BenchmarkExperiment ()
* Add root validator to BoxProp ()
* Add unit test for BenchmarkExperiment ()
* Added 2D support to get_observation() ()
* Added BoxProp min() and max(), ukpdated create_converted_space() ()
* Added comments ()
* Added list of lists support ()
* Added units GetStrFromUnit ()
* Added validation of consistency ()
* Added validator for BoxProp.unit ()
* Added validator for ObserveSensorValidator.output_units ()
* Bug fix in min()/max() ()
* Bug fixes and mypy fixes ()
* Change from numbers.Real to float ()
* Change type ()
* Changed typing of output_units ()
* Corrected FuelProp unit from percent to fraction ()
* Create MachSpeed units ()
* Created MachSpeed units ()
* Fix BenchmarkExperiment for new interface ()
* Fix validator ()
* Fixed unit representation issue ()
* Merge branch '67-observesensor-output-units' into 'main' (), closes  
* mypy ()
* mypy changes ()
* observation_space() uses create_converted_space() ()
* Remove default_min, default_max ()
* Removed type ignore ()
* Removed unneeded min()/max() ()
* Renamed Percent to PartOfWhole, added fraction ()
* Replaced list with abc.Sequence ()
* Update shape typing ()
* Updated output_units validation ()
* Updated units and spaces ()

# 1.38.0

### Bug Fixes

* Moved reset from ScriptedAction policy to CustomPolicy ()

### Features

* enforce an agent/platform naming convention ()

# 1.37.0

### Bug Fixes

* fixed configs to work with new interface ()
* interface issue with agent validation test ()

### Features

* add exclusiveness checking to controllers ()
* running successfully with multiple agents per platform ()

## 1.36.2

### Bug Fixes

* Merge remote-tracking branch 'origin/main' into docking1d ()
* updates for orginizatio, linting and testing ()

## 1.36.1

### Bug Fixes

* fixed issue with function call arg vs kwarg ()
* issue with indexing of box spaces ()
* made normalization automatic; fixed raw_obs issue ()
* removed )
* removed override annotation ()
* updates to allow push to other repo - temp fix --- also fix the linting ()

# 1.36.0

### Bug Fixes

* change wrap_dict to wrapped ()
* test pass for dict wrapper glue ()
* typing and formatting pipeline fails ()
* typos ()

### Features

* functor and base glue/reward/done ()

## 1.35.1

### Bug Fixes

* updates for the code server and test for other repo push ()

# 1.35.0

### Features

* rewrote obs extractor to simplify interface ()

## 1.34.4

### Bug Fixes

* update permissions for hpc image in rl-simulator ()

## 1.34.3

### Bug Fixes

* Merge branch 'main' of github.com/act3-ace:act3-rl/corl into main ()
* trying to get a new runner - updates docs ()

## 1.34.2

### Bug Fixes

* **dca:** added a unique_name field to TargetValueDifference. ()
* Merge remote-tracking branch 'origin/main' into dca-1v1 ()
* update to correct mkdocs?? ()
* updates for lint ()
* updates for lint and just added docs updates ()
* updates to get docs ()

## 1.34.1

### Bug Fixes

* fix Nonetype and None used in required units causing issues ()

# 1.34.0

### Features

* MultiMeasurementOperation class ()

# 1.33.0

### Bug Fixes

* cleaned up scripted action policy ()
* fixed issue where the default action wasn't handled properly in conjunction with multiple agents ()
* forgot to add scripted_action test policy ()
* modified 'controllers' definition to not use a dict ()
* reverted changes w.r.t. disabling normalization via the policy_config ()
* updated and ensured that it also works with the cartpole env ()

### Features

* (wip) added scripted_action policy ()

## 1.32.1

### Bug Fixes

* **Dockerfile:** rest permissions on /tmp ()
* **Dockerfile:** rest permissions on /tmp ()

# 1.32.0

### Bug Fixes

* Added units class to list ()

### Features

* Adds plumbing for KIAS sensor ()

## 1.31.5

### Bug Fixes

* change to individual dep images ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* **Dockerfile:** add sources.list and sources.list.d ()
* remove focal ()
* remove focal ()
* remove focal ()
* removing apt mirror for now ()
* revert agents base version and remove chrome ()
* sources.list changes ()
* sources.list changes ()
* sources.list changes ()
* update apt cache ()
* update apt-mirror-url var ()
* update apt-mirror-url var ()
* update apt-mirror-url var ()
* update external deps ()
* update external deps ()
* update external deps ()
* update external deps ()
* update sources.list ()
* update sources.list ()
* update sources.list ()
* update tag ()
* updates for e4xternal deps ()
* updates for external deps ()
* updates for external deps ()

## 1.31.4

### Bug Fixes

* Eliminate delete before use with done platforms ()

## 1.31.3

### Bug Fixes

* cleaned

## 1.31.2

### Bug Fixes

* 1. update requirements, 2. Make dict mapping, 3. add requirement ()
* add missing commit and fix test ()
* enable duplicate parts via unique part names ()
* fix class name ()
* pin packages ()
* update ray ()

## 1.31.1

### Bug Fixes

* move to latest code server 4.0.2 - tested local ()

# 1.31.0

### Features

* make curl stage external container ()

## 1.30.4

### Bug Fixes

* add reward_wrapper base classes ()
* added wrappers for dones and rewards, updated base agent to support ()

## 1.30.3

### Bug Fixes

* cleanup to remove references to items ()
* cleanup to remove references to items ()
* cleanup to remove references to items ()
* linting ()
* Merge branch 'cleanup/references_sims' into 'main' ()

## 1.30.2

### Bug Fixes

* Fixed episode_id ()
* Updated unit test ()
* Updated unit test ()

## 1.30.1

### Bug Fixes

* Fixed precision warning ()
* Merge branch 'precision-warning' into 'main' ()
* Removed np.array ()

# 1.30.0

### Bug Fixes

* Update embed_propreties to new variable names ()

### Features

* Add TAS/CAS units ()
* Merge branch 'tas_cas_unit' into 'main' ()

## 1.29.8

### Bug Fixes

* changed typing ()
* Incorporate changes from act3 agents ()
* Merge branch 'delete-moved-controller-wrappers' into 'main' ()

## 1.29.7

### Bug Fixes

* Changed argument names ()
* Merge branch '49-baseplatformpart-validator' into 'main' (), closes
* Removed pylint disable ()
* Rename properties to property_class ()

## 1.29.6

### Bug Fixes

* add BoxProp2D and update nan_check ()
* **boxprob2d:** update observe sensor glue to handle BoxProp2D ()
* **boxprob2d:** update sensors)
* **boxprop2d:** added boxprop2d to handle NetHack env ()
* **test:** add test for 2d boxprop ()
* update types for mypy ()

## 1.29.5

### Bug Fixes

* Exclusiveness as property ()
* Merge branch '48-exclusiveness-as-property' into 'main' (), closes
* Removed comment ()
* Removed unneeded inits ()
* Removed unneeded property ()

## 1.29.4

### Bug Fixes

* Merge branch 'simulator-validators-as-properties' into 'main' ()
* More validators as properties ()
* Updated OpenAIGymSimulator init ()

## 1.29.3

### Bug Fixes

* Added asserts ()
* Added asserts ()
* Merge branch '47-validators-as-properties' into 'main' (), closes
* Validators as properties ()

## 1.29.2

### Bug Fixes

* Method to embed properties ()
* Update controllers and platform parts also ()

## 1.29.1

### Bug Fixes

* Added env end_episode_on_first_agent_done config option ()
* Changes to support deleting platforms from simulation ()
* pylint ()
* Removing _set_all_done() ()

# 1.29.0

### Features

* Add controller wrappers ()
* Added RllibExperiment get_callbacks method ()

# 1.28.0

### Bug Fixes

* Raise instance not class ()

### Features

* Add episode reward ()
* Merge branch 'move_episode_reward_to_core' into 'main' ()

## 1.27.2

### Bug Fixes

* Make plugin library fail on partial incompatible match ()

## 1.27.1

### Bug Fixes

* updates to controller to report invalid state correctly for non len 1 arrays ()

# 1.27.0

### Bug Fixes

* fix trainable class check ()
* merge in the master to branch ()

### Features

* avoid adding policies for agents not accessed by rllib ()

## 1.26.15

### Bug Fixes

* cleanup to fix repeated field convert ()
* cleanup to fix repeated field convert ()

## 1.26.14

### Bug Fixes

* clean up linting ()
* small updates for code to support HLP - small cleanup i ()

## 1.26.13

### Bug Fixes

* clean up linting ()
* updates to add in profile examples ()
* updates to make the pylint items identifiable and to add in profiling ()

## 1.26.12

### Bug Fixes

* update to allow reset of agent config as part of reset of simulator ()

## 1.26.11

### Bug Fixes

* updates for the --- TEMP until we redesign restart ()

## 1.26.10

### Bug Fixes

* updates so that aers go to the same dir every time --- skip the date and pbs keys ()

## 1.26.9

### Bug Fixes

* updates to the code base to allow skipping setup of the simulator on reinit. ()

## 1.26.8

### Bug Fixes

* move to use foreach call ()
* update linting... ()

## 1.26.7

### Bug Fixes

* Add debug flag ()
* Add help message to debug flag ()
* Merge branch '43-add-debug-flag-to-train_rl' into 'main' (), closes

## 1.26.6

### Bug Fixes

* **simulator:** fix the environment not passing worker index and vector index to simulator correctly ()

## 1.26.5

### Bug Fixes

* Log at bound successes ()

## 1.26.4

### Bug Fixes

* Add variable logging ()

## 1.26.3

### Bug Fixes

* Error checking, horizon from rllib_config ()

## 1.26.2

### Bug Fixes

* Add data parameter checking for TabularParameterProvider ()
* Merge branch 'evaluation-param-check' into 'main' ()

## 1.26.1

### Bug Fixes

* Add support for calling method on remote epp ()
* Merge branch 'evaluation-remote-support' into 'main' ()

# 1.26.0

### Bug Fixes

* Add string value support to ConstantParameter ()
* Improved TabularParameterProvider validators ()
* Removed comment ()
* TabularParameterProvider related typing ()

### Features

* Add support for evaluation ()
* Add TabularParameterProvider ()
* Merge branch 'evaluation-support' into 'main' ()

## 1.25.8

### Bug Fixes

* the following fixes formatting... ()
* update to get box prop working ()

## 1.25.7

### Bug Fixes

* fix value error from nvidia-smi being on the cpu nodes ()
* Merge branch 'fix/nvidia-smi-resource-error-cpu' into 'main' ()

## 1.25.6

### Bug Fixes

* revert back to tf 2.4.0 given tcn issues ()
* revert back to tf 2.4.1 given tcn issues ()
* try again to use agent base 2.2.1 ()

## 1.25.5

### Bug Fixes

* hot fix to update epp parameter signature in experiment to match for save and load ()
* hot fix to update epp parameter signature in experiment to match for save and load ()
* hot fix to update epp parameter signature in experiment to match for save and load ()

## 1.25.4

### Bug Fixes

* **callbacks:** match default callbacks to new signature ()
* **ray:** update ray to 1.9 ()
* updated epp access to be from the config ()
* updated key from glues->dones ()

## 1.25.3

### Bug Fixes

* add back policy ()
* lint fix: ()

## 1.25.2

### Bug Fixes

* small changes ()

## 1.25.1

### Bug Fixes

* ignore new mypy version errors ()
* updates to tensorflow 2.7.0 cuda 11.1 torch 1.8 ()

# 1.25.0

### Features

* add no op agents and controllers ()

# 1.24.0

### Bug Fixes

* Add EPP checkpoint ()
* BoundStepUpdater reverse bounded by original value ()
* Connect EPP metrics in callback ()
* Enable agent replacement for agent EPP configuration ()
* Remove debugging print statements ()
* Use new buffer size variable ()

### Features

* Automatic Domain Randomization ()

## 1.23.2

### Bug Fixes

* change all parts to accept platform, config, prop ()

## 1.23.1

### Bug Fixes

* add missing **init**.py ()

# 1.23.0

### Features

* add non-trained policy support for rllib experiments ()

# 1.22.0

### Features

* **dones:** added shared dones, fixed aer strings ()

## 1.21.2

### Bug Fixes

* **warning:** Fix precision warning ()
* **warning:** Fix precision warning ()
* **warning:** Fix precision warning ()

## 1.21.1

### Bug Fixes

* updates to add obs to the apply at agent level ()

# 1.21.0

### Bug Fixes

* Ignore pylint error ()
* Remove position hack ()

### Features

* Enable EPP for simulation reset ()

## 1.20.1

### Bug Fixes

* **configs:** change default configs to ignore_reinit_error True ()

# 1.20.0

### Bug Fixes

* Don't call EPP.get_params from init ()
* EPP.get_params validation ()
* Parameter provider tests and fixes ()
* Remove unused pydantic models ()

### Features

* Agent episode parameter providers ()
* Environment episode parameter providers ()
* Registry ()
* Remote support for episode parameter providers ()

## 1.19.2

### Bug Fixes

* Add episode length done automatically ()
* Prohibit EpisodeLengthDone in agent dones ()

## 1.19.1

### Bug Fixes

* Create environment on driver in rllib experiment ()

# 1.19.0

### Features

* **glue:** added glue multi wrapper base class ()

# 1.18.0

### Features

* add non trainable base ()

# 1.17.0

### Bug Fixes

* **rewards:** fixed post_process_trajectory reward function signature ()

### Features

* **rewards:** implemented post_process_trajectory call ()

## 1.16.2

### Bug Fixes

* **dones:** added shared done base class ()

## 1.16.1

### Bug Fixes

* nan check applied controls ()
* rename method ()
* update typing ()

# 1.16.0

### Bug Fixes

* Change functor from iterator to sequence. ()
* Default task dones in environment variable store ()
* Move factory to separate library ()
* Move functors to library ()
* Rename environment glue/reward/done creation methods ()

### Features

* Environment parameters passed to agent ()
* Pydantic checking of ACT3MultiAgentEnv ()

# 1.15.0

### Bug Fixes

* Change automatic name for SensorBoundsCheckDone ()
* Don't fail on redundant functor names if no parameters ()
* **gitignore:** Added ray_logs to gitignore ()
* **gitignore:** Added vscode directory to gitignore ()
* **parameter:** Fix issue with test_choices_string() ()
* **parameter:** Force return of native types ()
* **parameter:** Removed get_native() method, now automatically handled by units class ()
* pylint fix ()
* Replace constr with Annotated ()
* Start on glues and rewards ()
* **units:** Added strict types and fixed handling of none unit ()
* **units:** Added value validator to automatically convert numpy types ()

### Features

* Better reference handling ()
* Connect dones to new functor argument ()
* Merge branch '7-agent-config-variable-store' into 'main' (), closes
* **parameter:** Switch to using the core Parameter class ()
* Sensor bounds done and delayed unit resolution ()
* **test:** Added units unit test ()
* Unit conversion and storage parameters in Functor ()

## 1.14.1

### Bug Fixes

* **gitlab-ci:** Change stages ()

# 1.14.0

### Features

* adds repeated observe sensor and supports openaigym repeated ()

## 1.13.1

### Bug Fixes

* **vscode:** fix debugging in vscode ()

# 1.13.0

### Features

* **plugin library:** update how plugin library works/extracts ()

## 1.12.2

### Bug Fixes

* check array shape ()

## 1.12.1

### Bug Fixes

* **parameter:** Add some typing ()
* **parameter:** Make get_validator() not abstract ()
* **parameter:** Relocated bound_func attribute ()
* Rework types for pylint ()

# 1.12.0

### Features

* Merge branch '17-update-ray-to-version-1.7' into 'main' (), closes

# 1.11.0

### Features

* Pydantic validation for environments ()

## 1.10.6

### Bug Fixes

* bad key name ()

## 1.10.5

### Bug Fixes

* Merge branch 'fix/setup.py' into 'main' ()
* updates to make sure the setup.py and the requirements.txt are synced ()

## 1.10.4

### Bug Fixes

* use policy config ()

## 1.10.3

### Bug Fixes

* Merge branch 'fix/update_correct_names' into 'main' ()
* updates to remove 2 containers ()

## 1.10.2

### Bug Fixes

* Merge branch 'fix/update_correct_names' into 'main' ()
* update to correct names for targets in .gitlab-ci.yml ()

## 1.10.1

### Bug Fixes

* added missing item to docs ()
* Merge branch 'fix/package2' into 'main' ()

# 1.10.0

### Features

* add container stages for release ()
* Merge branch 'fix-container-package' into 'main' ()

# 1.9.0

### Features

* fix package ()
* Merge branch 'fix/package2' into 'main' ()

# 1.8.0

### Features

* plumbing for aer file writing ()

# 1.7.0

### Bug Fixes

* force mypy to pass in ci ()
* use pytest ()

### Features

* add me parts test ()

## 1.6.2

### Bug Fixes

* Fix versioning problems for packages. Updates to docker to support development environments. ()

## 1.6.1

### Bug Fixes

* updates to add in container building ()
* updates to add in vscode builds ()

# 1.6.0

### Bug Fixes

* Fix header ()

### Features

* Add EpisodeLengthDone, SensorBoundsCheckDone ()
* Utilize validators for done conditions ()

# 1.5.0

### Features

* added in configuration to enable per-agent policy configurations ()

## 1.4.2

### Bug Fixes

* Update Dockerfile ()

## 1.4.1

### Bug Fixes

* Update Dockerfile ()
* Update Dockerfile ()
* Update Dockerfile ()

# 1.4.0

### Bug Fixes

* fixed dict -> ordereddict ()

### Features

* Added observation filtering ()

## 1.3.5

### Bug Fixes

* Update Dockerfile ()

## 1.3.4

### Bug Fixes

* **setup.py:** changes for MR ()
* **setup.py:** remove MIT line ()
* **setup.py:** remove package dir ()
* **setup.py:** remove package dir ()
* **setup.py:** update for pip installs ()
* **setup.py:** update for pip installs ()

## 1.3.3

### Bug Fixes

* Add in missing init file ()

## 1.3.2

### Bug Fixes

* Update setup.py ()

## 1.3.1

### Bug Fixes

* updates to the build number version ()

# 1.3.0

### Bug Fixes

* add base props ()
* bad merge ()
* clean up 6dof after merge ()
* clean up base properties ()
* dtype default ()
* exception text ()
* fix BaseModel derivations ()
* linting ()
* linting ()
* mypy ()
* mypy ()
* remove MultiBox ()
* rename file ()
* spelling ()
* style ()
* update doc strings ()
* validate control against gym space ()

### Features

* add tests ()
* define spaces as props held by parts ()

# 1.2.0

### Features

* move to ME parts to platform ()

# 1.1.0

### Bug Fixes

* adding plugin paths ()
* BaseTimeSensor ()
* default exclusiveness to None ()
* default exclusiveness to None ()
* **dockerfile:** update pip install ()
* Fix imports ()
* fix linting ()
* **gitlab-ci:** allow mkdocs to fail ()
* **gitlab-ci:** missed a plugin ()
* **gitlab-ci:** update mkdocs installs ()
* Glue extractors ()
* Platform utilities ()
* Platform utilities actually used ()
* Put plugin path in cartpole configuration ()
* remove learjet base controllers ()
* remove learjet configs ()
* Remove MainUtil dependency ()
* remove prop validation ()
* Remove unused import ()
* **requirements:** adrpy missing ()
* **requirements:** update imports ()
* Reward path ()
* Rewrite directory handling to avoid library module ()
* **setup.py:** trying the old one with a mod ()
* **setup.py:** update to fix import errors ()
* **setup:** minor errors ()
* Silly bugs ()
* Units and aero library ()
* Update plugin library dependency ()
* updates required for extension repo ()

### Features

* add plugin lib ()
* Initial copy from other repo act3 core ()

# 1.0.0 (2021-10-05)

### Bug Fixes

* add cicd ()
* add more files ()
* **gitlab-ci:** change to the pypi ()
* pytlint ()
* trying a differetn url ()
* update branch ()
* update pip_index_url ()
* update syntax and dockerfile ()
