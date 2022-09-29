Intitial release of CoRL - Part #1 -Approved on 2022-05-2024 12:08:51 - PA Approval # [AFRL-2022-2455]"

# ACT3 RL Core

Core act3 reinforcement learning library

<!-- - Online documentation: [https://act3-rl.git.act3-ace.com/corl/](https://act3-rl.git.act3-ace.com/corl/) -->

### How to install pip package from project level with a personal access token

In your User Settings->Access Tokens, you are able to create a Personal Access Token. Create a token with read_api scope. 
The name of the token will be your username, the value of the token will be your password. Be sure to save them, because you 
will not be able to view them again. 

Use this command to install the pip package:

```bash
pip install corl --no-deps --index-url https://<username>:<password>@git.act3-ace.com/api/v4/projects/657/packages/pypi/simple`
```

The `--no-deps` tag`     is optional. Use it if you do not want dependency packages installed with the package. The project should already have the dependencies in the package.

Do NOT use the command provided by GitLab that utilizes the `--extra-index-url tag`. This tag will check PYPI.org first for the package, 
and you will get an error.

## How to build the docker containers

The following project support developement via docker conatiners in vscode and on the DOD HPC. This is not strictly required but does provide the mode convient way to get started. ***Note:*** fuller documentation is availible in the documentation folder or online docs. 

- ***Setup the user env file:*** in code directory run the following script  --> `./scripts/setup_env_docker.sh`
- ***Build the docker containers using compose:*** run the following command --> `docker-compose build`


## How to build the documentation locally

This repository is setup to use [MKDOCS](https://www.mkdocs.org/) which is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file. Start by reading the introductory tutorial, then check the User Guide for more information.

- ***Install Mkdocs Modules*** in container/virtual environment run the following command --> `pip install -U -r mkdocs-requirements.txt`
- ***Build Documentation:*** Inside docker container run the following command --> `python -m  mkdocs build`
- ***Serve Documentation:*** Inside docker container run one of the following commands --> 
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
        - jamees.patrick@us.af.mil
    - Engineering Support
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
