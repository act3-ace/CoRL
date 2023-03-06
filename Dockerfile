# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

#########################################################################################
# Dependent Tags
#########################################################################################

ARG OCI_REGISTRY=reg.git.act3-ace.com

ARG PROJ_PKG=/opt/project/corl
ARG PKG_ROOT=/opt/libcorl
ARG AGENT_BASE_VERSION=v2.4.19

ARG AGENT_BASE_IMAGE_PATH=/act3-rl/agents-base/releases
#########################################################################################
# Utility image for combining the various package deps
# Purpose: This image essentially only takes the whl files from other images and puts in
#          one dependicy image. Goal is to reduce the copy lines in the Dockerfile
# 
# -------- > HACK FOR THE HPC!!!! < --------
#########################################################################################
FROM ${OCI_REGISTRY}${AGENT_BASE_IMAGE_PATH}/pytorch-rl-base:${AGENT_BASE_VERSION} AS whl_dependencies_hack

WORKDIR /opt/project

RUN echo "DEPS PATH ${DEPS_PATH}" \
    && mkdir -p /opt/dependencies

# CAN ADD DOCKER includes here

#########################################################################################
# Develop - Thin image that only contains the bare min deps for running
#########################################################################################
FROM ${OCI_REGISTRY}${AGENT_BASE_IMAGE_PATH}/pytorch-rl-base:${AGENT_BASE_VERSION} AS develop

WORKDIR /opt/project

COPY --from=whl_dependencies_hack /opt/dependencies /opt/dependencies

COPY . .

RUN poetry config virtualenvs.create false \
  && poetry install --with lint --without test,docs,profile,torch \
  && ray disable-usage-stats

#########################################################################################
# Code Server
#########################################################################################
FROM ${OCI_REGISTRY}${AGENT_BASE_IMAGE_PATH}/pytorch-rl-coder-user:${AGENT_BASE_VERSION} AS coder-user

ARG TARGET_USER=act3
ENV TARGET_USER=${TARGET_USER}
WORKDIR /opt/project

COPY --from=whl_dependencies_hack /opt/dependencies /opt/dependencies

COPY . .
USER root
RUN poetry config virtualenvs.create false \
  && poetry install --without torch \
  && ray disable-usage-stats 

USER ${TARGET_USER}:${TARGET_USER}

#########################################################################################
# Code Server
#########################################################################################
FROM ${OCI_REGISTRY}${AGENT_BASE_IMAGE_PATH}/pytorch-rl-coder-hpc:${AGENT_BASE_VERSION} AS coder-hpc

WORKDIR /opt/project

COPY --from=whl_dependencies_hack /opt/dependencies /opt/dependencies

COPY . .
USER root
RUN poetry config virtualenvs.create false \
  && poetry install --without torch \
  && ray disable-usage-stats

#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop AS build
ARG PKG_ROOT=/opt/libcorl
ENV PKG_ROOT=${PKG_ROOT}

WORKDIR /opt/project

COPY --from=whl_dependencies_hack /opt/dependencies /opt/dependencies

COPY . .

RUN poetry build -n  && mkdir ${PKG_ROOT} && mv dist/ ${PKG_ROOT}

#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the package stage contains everything required to install the project from another container build
# NOTE: a kaniko issue prevents the source location from using a ENV variable. must hard code path
FROM scratch AS package
COPY --from=build /opt/libcorl /opt/libcorl

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage AS cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.
FROM develop AS cicd
RUN poetry install --only test,docs,lint
RUN apt-get update -y && apt-get -y install git
