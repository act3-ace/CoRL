#!/bin/bash -x
# -------------------------------------------------------------------------------
#
#
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------# #######################################################################################
# Author: Benjamin Heiner
# Purpose: Sets up the docker environment file for build process
# #######################################################################################

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

# parse arguments below
DOCKER_CONTAINER=""
VDL_USER=""
CONTAINER_USER="act3rl"
BRANCH="master"
DOCKER_ONLY=0
DATE="$(date +"%m-%d-%y:%H%M")"
SHM_SIZE=6g

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# SETUP SOME OF THE PATHS NEEDED
# CODE_PATH is the directory that holds your cloned repository.
# eg. if you cloned your repo to /home/me/gitrepos/act3-rllib-agents
# You need to run this script from /home/me/gitrepos/act3-rllib-agents
# and that will be $CODE_PATH
CODE_PATH=$(realpath "$SCRIPT_DIR"/..)

# DEV_PATH is the location in the repo that holds the Dockerfile and docker-compose.yml
DEV_PATH="$CODE_PATH"/.devcontainer
# DATA_PATH is where you want ray to write all the stuff it writes to disk.
# eg. if you want ray to write results to /home/me/ray_results
# Then DATA_PATH should should be /home/me
DATA_PATH="$HOME"/data

# Linux  Terminal
mkdir -p "$DEV_PATH"/zdl_forbuilds   #this is where you are going to download the stuff in the next step
mkdir -p "$DATA_PATH"/ray_results

# Linux  Terminal
#export UID="$(id -u)"  This should be a set reserved variable and so don't set
GID="$(id -g)"
GROUP="$(id -gn)"
NEW_UID=$UID  #Change these if you want to specify a different UID/GID for the user in the container
NEW_GID=$GID  #Change these if you want to specify a different UID/GID for the user in the container
NEW_USER=$CONTAINER_USER
NEW_GROUP=$CONTAINER_USER

cd "$CODE_PATH" || { echo "Failure. Could not change directory to ${CODE_PATH}"; exit 1; }
COMPOSE_FILE="$CODE_PATH"/docker-compose.yml

USE_ACT3_OCI_REGISTRY=${ACT3_OCI_REGISTRY:-reg.git.act3-ace.com}
USE_OCI_REGISTRY=${OCI_REGISTRY:-reg.git.act3-ace.com}
USE_AGENTS_BASE_IMAGE=${AGENTS_BASE_IMAGE:-nvcr.io/nvidia/pytorch:22.05-py3}
USE_DOCKER_OCI_REGISTRY=${DOCKER_OCI_REGISTRY:-reg.git.act3-ace.com/act3-rl/external-dependencies/}

ENV_FILE="$CODE_PATH"/.env
{
  echo '# This .env file is used by docker-compose to substitute in environment variables in the docker-compose.yml'
  echo '# Double quotes on the code and data paths are not needed for this file.'
  echo "CODE_PATH=${CODE_PATH}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "UID=${UID}"
  echo "GID=${GID}"
  echo "NEW_UID=${NEW_UID}"
  echo "NEW_GID=${NEW_GID}"
  echo "USER=${USER}"
  echo "GROUP=${GROUP}"
  echo "NEW_USER=${NEW_USER}"
  echo "NEW_GROUP=${NEW_GROUP}"
  echo "BRANCH=${BRANCH}"
  echo "DATE=${DATE}"
  echo "SHM_SIZE=${SHM_SIZE}"
  echo "COMPOSE_DOCKER_CLI_BUILD=1"
  echo "DOCKER_BUILDKIT=1"
  echo "DOCKER_CLI_EXPERIMENTAL=enabled"
  echo "ACT3_OCI_REGISTRY=${USE_ACT3_OCI_REGISTRY}"
  echo "AGENTS_BASE_IMAGE=${USE_AGENTS_BASE_IMAGE}"
  echo "OCI_REGISTRY=${USE_OCI_REGISTRY}"
  echo "DOCKER_OCI_REGISTRY=${USE_DOCKER_OCI_REGISTRY}"

} >| "${ENV_FILE}"

# rsync -a --delete ~/.ssh/ "$DEV_PATH"/.ssh/

echo "done"
