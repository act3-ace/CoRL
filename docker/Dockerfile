# -------------------------------------------------------------------------------
# 
#
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# ROOT USER ITEMS
# -------------------------------------------------------------------------------
FROM python:3.10.11-slim-bullseye as corl-base
ARG NEW_GID=1000
ARG NEW_UID=1000
ARG NEW_GROUP=developer
ARG NEW_USER=developer
ARG POETRY_VERSION=1.8.2

RUN apt-get update -y && \
  apt-get install -y --no-install-recommends \
  apt-utils \
  ca-certificates \
  curl \
  dumb-init \
  git \
  git-lfs \
  htop \
  ssh \
  nano \
  openssh-client \
  patch \
  procps \
  rsync \
  ssh \
  sudo \
  vim \
  zsh \
  lsb-release \
  libsecret-1-0 \
  x11-apps && \
  git lfs install &&\
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY ./docker/dle.crt /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

RUN \
  curl -sSL https://install.python-poetry.org | \
  POETRY_HOME=/usr/local python3 - --version $POETRY_VERSION

RUN \
  addgroup --gid $NEW_GID $NEW_GROUP && \
  useradd -m -l --uid $NEW_UID --gid $NEW_GID $NEW_USER && \
  usermod -aG sudo ${NEW_USER} && \
  echo "${NEW_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN mkdir -p /opt/data && \
  chmod -R 777 /opt/data

# -------------------------------------------------------------------------------
# DEVELOPER USER ITEMS
# -------------------------------------------------------------------------------
FROM corl-base as corl-dev
ARG NEW_GID=1000
ARG NEW_UID=1000
ARG NEW_GROUP=developer
ARG NEW_USER=developer

USER $NEW_USER:$NEW_GROUP
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV VENV_PATH=/home/$NEW_USER/virtualenvs
ENV PATH="$VENV_PATH/bin:$PATH" VIRTUAL_ENV=$VENV_PATH
ENV WORKDIR=/opt/project
ENV SRCDIR=/opt/project
WORKDIR ${WORKDIR}

RUN python -m venv $VENV_PATH

COPY pyproject.toml poetry.lock ${WORKDIR}/

RUN \
 --mount=type=secret,id=poetry_auth,dst=/home/$NEW_USER/.config/pypoetry/auth.toml,uid=$NEW_UID \
 poetry install --no-root --no-cache --no-interaction && \
 rm -rf /home/$NEW_USER/.cache/pypoetry/*


# -------------------------------------------------------------------------------
# HPC
# -------------------------------------------------------------------------------
FROM corl-dev as corl-hpc

# Paths for HPCs
USER root

RUN	mkdir -p /usr/local/Modules\
  /external_bin \
  /external_lib \
  /p \
  /p/work1 \
  /p/work2 \
  /p/work3 \
  /p/app \
  /work  \
  /workspace \
  /app \
  /apps \
  /app/projects \
  /opt/cray \
  /usr/cta \
  /usr/cta/unsupported \
  /usr/share/Modules \
  /opt/modules \
  /opt/cray/pe/ \
  /etc/opt/cray \
  /cm

RUN curl -fsSL https://code-server.dev/install.sh | sh

COPY scripts/codeserver scripts/codeserver
RUN ./scripts/codeserver/install_code_server_plugins.sh
RUN ln -s /usr/bin/code-server /usr/local/bin/code-server

ARG NEW_USER=developer
ENV VENV_PATH=/home/$NEW_USER/virtualenvs
ENV PATH="$VENV_PATH/bin:$PATH" VIRTUAL_ENV=$VENV_PATH
ENV WORKDIR=/opt/project
ENV SRCDIR=/opt/project
ENV ACT3_CODE_PATH=/opt/project
ENV AI_ARTIFACTS_FOLDER=/opt/data
WORKDIR ${WORKDIR}
RUN python -m venv $VENV_PATH

EXPOSE 8888
ENTRYPOINT ["scripts/codeserver/entrypoint.sh"]
