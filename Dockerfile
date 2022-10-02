# ---------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning (RL) Core.
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# ---------------------------------------------------------------------------

ARG AGENTS_BASE_IMAGE=nvcr.io/nvidia/pytorch:22.01-py3
ARG CODE_VERSION=4.6.1
# set pip environment variable to disable pip upgrade warning
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

######################################
# Curl Base
######################################
FROM alpine as curl-base

# install curl
RUN apk update \
    && apk --no-cache add\
        curl \
    && rm -rf /var/cache/apk/*

# Download external files and save to /opt/temp
WORKDIR /opt/temp

######################################
# fixuid-0.4-linux-amd64.tar.gz
######################################
FROM curl-base as fixuid-base
RUN curl -sSLO https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz && \
    mkdir fixuid && \
    tar -xzf /opt/temp/fixuid-0.4-linux-amd64.tar.gz -C fixuid

FROM scratch as fixuid
COPY --from=fixuid-base /opt/temp/fixuid /opt/temp/fixuid

####################################### 
# code-server_${CODE_VERSION}_amd64.deb
#######################################
FROM curl-base as vs-code-server-base
ARG CODE_VERSION
RUN curl -sSLO https://github.com/cdr/code-server/releases/download/v${CODE_VERSION}/code-server_${CODE_VERSION}_amd64.deb 

FROM scratch as vs-code-server
ARG CODE_VERSION
COPY --from=vs-code-server-base /opt/temp/code-server_${CODE_VERSION}_amd64.deb /opt/temp/

#######################################
# google-chrome-stable_current_amd64.deb
#######################################
FROM curl-base as chrome-base
RUN curl -SsLO https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 

FROM scratch as chrome
COPY --from=chrome-base /opt/temp/google-chrome-stable_current_amd64.deb /opt/temp/

#######################################
# DuoConnect-latest.tar.gz
#######################################
FROM curl-base as duo-connect-base
RUN curl -SsLO https://dl.duosecurity.com/DuoConnect-latest.tar.gz && \
    mkdir duoconnect && \
    tar xvzf /opt/temp/DuoConnect-latest.tar.gz -C duoconnect

FROM scratch as duo-connect
COPY --from=duo-connect-base /opt/temp/duoconnect /opt/temp/duoconnect

#######################################
# bash-git-prompt
#######################################
FROM curl-base as bash-git-prompt-base
RUN curl -SsLO https://github.com/magicmonty/bash-git-prompt/archive/refs/tags/2.7.1.tar.gz && \
    mkdir bash-git-prompt && \
    tar xvzf /opt/temp/2.7.1.tar.gz -C bash-git-prompt

FROM scratch as bash-git-prompt
COPY --from=bash-git-prompt-base /opt/temp/bash-git-prompt /opt/temp/bash-git-prompt


#########################################################################################
# develop stage contains base requirements. Used as base for all other stages.
# builds on the base CUDA image used by other prjections.
#
#   (1) APT Install deps for the base development only - i.e. items for running code
#   (2) Install the repository requirements
#   (3) logs file created
#
#########################################################################################
FROM ${AGENTS_BASE_IMAGE} as develop

# FIX NVIDIA CONTAINER ISSUE
RUN rm /etc/apt/sources.list.d/cuda.list || continue && rm /etc/apt/sources.list.d/nvidia-ml.list || continue && apt-key del 7fa2af80

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # for graph plotting with keras
        graphviz \
        # for plotting with matplotlib
        python3-tk \
        # ray requirements
        rsync \
        zlib1g-dev \
        libgl1-mesa-dev \
        libgtk2.0-dev \
        cmake \
        # to install requirements via git
        git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY poetry.lock pyproject.toml ./

RUN /opt/conda/bin/python -m pip install poetry==1.2.1

# Docker is isolating the env --- bypass virtual env in poetry
# alternate would be to use multi stage and copy env
# TODO poetry install will all the packages, let's min install later
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi
RUN ray disable-usage-stats

RUN mkdir -p /opt/data/corl/run_logs/

#########################################################################################
# Build stage packages from the source code
#########################################################################################
FROM develop as build
ENV CORL_ROOT=/opt/libcorl
ARG PIP_INDEX_URL

WORKDIR /opt/project
COPY . .

RUN poetry build -n && mv dist/ ${CORL_ROOT}
#########################################################################################
# OpenGL Base for the visulization and other items
# !!This ensures that the base cuda image has OPENGL support for items to X11 forward
#########################################################################################
FROM develop as glbase-builder

RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 \
        libxdmcp6 \
        libxcb1 \
        libxext6 \
        libx11-6 \
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 && \
        rm -rf /var/lib/apt/lists/*

COPY .devcontainer/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

##########################################################################################
# VERSION - Makes version file to be copied in the target stage
##########################################################################################
FROM busybox as version

RUN mkdir /etc/act3
WORKDIR /etc/act3

ARG VERSION
ARG GIT_COMMIT_HASH
ARG PROJECT_NAME

RUN echo -e " NAME=${PROJECT_NAME}\n \
VERSION=${VERSION}\n \
GIT COMMIT HASH=${GIT_COMMIT_HASH}" \
> act3_version.txt

##########################################################################################
# USER BUILDER - Setup users for builders...
#
# This is the top level of all containers. Mainly looking to ensure that we have user setup
# for all of stages in build proecess.
#
# 1. Removes need to convert post steps
# 2. Single location for items
##########################################################################################
FROM glbase-builder AS user-builder

ARG NEW_GID=1000
ARG NEW_UID=1000
ARG NEW_GROUP=act3rl
ARG NEW_USER=act3rl

ENV NEW_GID=$NEW_GID
ENV NEW_UID=$NEW_UID
ENV NEW_GROUP=$NEW_GROUP
ENV NEW_USER=$NEW_USER

# Install basic utilities
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --gid $NEW_GID $NEW_GROUP \
    && useradd -m -l -s /bin/bash -u $NEW_UID -g $NEW_GID $NEW_USER \
    && usermod -aG sudo ${NEW_USER} \
    && echo "${NEW_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

##########################################################################################
# ROOT ACT3 RL CORE DEVELOPMENT BASE
##########################################################################################
FROM user-builder AS base-builder

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

ARG NAMESPACE=base
ARG NEW_USER=act3rl

ENV CODE=/opt/project

WORKDIR /opt/temp

RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Basic Dependcies
RUN apt-get update --fix-missing \
    && apt-get --no-install-recommends install -y \
        # required for compiling/building code
        build-essential \
        # downloading tools
        git \
        curl \
        wget \
        # other tools
        bzip2 \
        libsm6 \
        pciutils \
        iputils-ping \
        apt-utils \
        ssh \
        htop \
        tmux \
        vim \
        cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /opt/temp

WORKDIR $CODE

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc"

# I this this is thr problem stage. Will test later. Adding to other stages to catch in all places
RUN chmod 1777 /tmp

##########################################################################################
# USER corl DEVELOPMENT BASE
##########################################################################################
FROM base-builder AS user-base-builder-x

ARG NEW_USER=act3rl

USER ${NEW_USER}

ENV CODE=/opt/project
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH ${CONDA_DIR}/bin:$PATH
ENV PYTHONPATH=${CODE}:$PYTHONPATH

# Setup up some familiar aliases for users
RUN echo "alias ls='ls --color=auto'" >> /home/${NEW_USER}/.bashrc \
    && echo "alias grep='grep --color=auto'" >> /home/${NEW_USER}/.bashrc \
    && echo "alias ll='ls -alF'" >> /home/${NEW_USER}/.bashrc \
    && echo "alias la='ls -A'" >> /home/${NEW_USER}/.bashrc \
    && echo "alias l='ls -CF'" >> /home/${NEW_USER}/.bashrc

# Install bash-git-prompt
COPY --from=bash-git-prompt /opt/temp/bash-git-prompt /home/${NEW_USER}/.bash-git-prompt
COPY .devcontainer/bash_update.sh /tmp/
RUN cat /tmp/bash_update.sh >> /home/${NEW_USER}/.bashrc


# Persist bash history between runs
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && sudo mkdir /commandhistory \
    && sudo touch /commandhistory/.bash_history \
    && sudo chown -R ${NEW_USER} /commandhistory \
    && echo $SNIPPET >> "/home/${NEW_USER}/.bashrc"

# Avoiding extension reinstalls on container rebuild (VSCODE)
RUN mkdir -p /home/${NEW_USER}/.vscode-server/extensions \
        /home/${NEW_USER}/.vscode-server-insiders/extensions \
    && chown -R ${NEW_USER} \
        /home/${NEW_USER}/.vscode-server \
        /home/${NEW_USER}/.vscode-server-insiders \
    && sudo chown -R ${NEW_USER} /home/${NEW_USER}

# RUN pre-commit install
COPY .devcontainer/startup.sh /tmp/startup.sh

# expose port for tensorboard
EXPOSE 6006

# setup fixuid to make container portable across users with differing uid/gid
USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        dumb-init \
        libtiff5 \
    && rm -rf /var/lib/apt/lists/*

# install fixuid
COPY --chown=${NEW_USER} --from=fixuid /opt/temp/fixuid /usr/local/bin/
RUN chown root:root /usr/local/bin/fixuid \
    && chmod 4755 /usr/local/bin/fixuid \
    && mkdir -p /etc/fixuid \
    && printf "user: ${NEW_USER}\ngroup: ${NEW_USER}\n" > /etc/fixuid/config.yml

# install duoconnect
COPY --from=duo-connect /opt/temp/duoconnect /tmp/
RUN cd /tmp && /tmp/install.sh
# reset permissions on /tmp, temp fix until culprit is found
RUN chmod 1777 /tmp

USER ${NEW_USER}

######################################################################################################################
# User Base builder container
######################################################################################################################
# Copy in the code at the end in another stage so
# future stages can build from user-base-builder-x w/o copying the code
FROM user-base-builder-x as user-base-builder

ARG NEW_USER=act3rl

# copy in the corl repo
COPY --chown=${NEW_USER} . $CODE

# Copy version file from version stage
COPY --from=version /etc/act3 /etc/act3

# save commit hash. used by the integration repo when automatically building images
ARG ACT3_RLLIB_AGENTS_GIT_COMMIT=unspecified
LABEL act3_rllib_agents_git_commit=$ACT3_RLLIB_AGENTS_GIT_COMMIT
RUN mkdir -p /home/${NEW_USER}/corl-info \
    && echo ${ACT3_RLLIB_AGENTS_GIT_COMMIT} >> /home/${NEW_USER}/corl-info/commit.txt

##########################################################################################
# through
##########################################################################################
FROM user-base-builder-x AS user-code-dev-x
ARG NEW_USER=act3rl
ENV CODE=/opt/project/

USER root
# reset permission on /tmp tp allow apt install, temp fix
RUN chmod 1777 /tmp
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        dumb-init \
        htop \
        locales \
        man \
        nano \
        git \
        procps \
        ssh \
        sudo \
        net-tools \
        vim \
        libtiff5 \
    && rm -rf /var/lib/apt/lists/*


# https://wiki.debian.org/Locale#Manually
RUN sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen \
    && locale-gen

ENV LANG=en_US.UTF-8
RUN chsh -s /bin/bash
ENV SHELL=/bin/bash

# Install code-server
ARG CODE_VERSION=4.6.1
COPY --chown=${NEW_USER} --from=vs-code-server /opt/temp/code-server_${CODE_VERSION}_amd64.deb /opt/deps/code-server_${CODE_VERSION}_amd64.deb
RUN apt-get install -y /opt/deps/code-server_${CODE_VERSION}_amd64.deb \
    && ln -s /usr/bin/code-server /usr/local/bin/code-server

RUN chown -R ${NEW_USER}:${NEW_USER} ${CODE}
#
# Add user to docker group
USER ${NEW_USER}
# RUN sudo usermod -aG docker ${NEW_USER}

RUN code-server --install-extension streetsidesoftware.code-spell-checker \
    && code-server --install-extension bierner.markdown-mermaid \
    && code-server --install-extension DavidAnson.vscode-markdownlint \
    && code-server --install-extension eamodio.gitlens \
    && code-server --install-extension ms-python.anaconda-extension-pack \
    && code-server --install-extension ms-python.python \
    && code-server --install-extension shd101wyy.markdown-preview-enhanced \
    && code-server --install-extension yzhang.markdown-all-in-one \
    && code-server --install-extension njpwerner.autodocstring \
    && code-server --install-extension tomoki1207.pdf \
    && code-server --install-extension auchenberg.vscode-browser-preview \
    && code-server --install-extension dracula-theme.theme-dracula


EXPOSE 8888
WORKDIR ${CODE}
ENTRYPOINT ["dumb-init", "fixuid", "-q", "code-server", "--host", "0.0.0.0", "--port", "8888", "--auth", "none", "."]

##########################################################################################
# user-code-dev
##########################################################################################
FROM user-code-dev-x as user-code-dev

ARG NEW_USER=act3rl

# copy in the corl repo
COPY --chown=${NEW_USER} . $CODE

# Copy version file from version stage
COPY --from=version /etc/act3 /etc/act3

#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################
# the package stage contains everything required to install the project from another container build
# NOTE: a kaniko issue prevents the source location from using a ENV variable. must hard code path
FROM scratch as package
ENV CORL_ROOT=/opt/libcorl
COPY --from=build ${CORL_ROOT} ${CORL_ROOT}

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
FROM build as cicd
# 