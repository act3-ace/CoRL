#!/bin/bash

# this script is called at dev container creation time and
# may be used to perform any custom setup, install, etc.
# that is not part of the docker build process

# conda init
source ~/.bashrc

# this is already done in the Dockerfile when the image is built
# git clone https://github.com/magicmonty/bash-git-prompt.git ~/.bash-git-prompt

# the following is done automatically by vscode remote containers extension when container is launched
# according to https://code.visualstudio.com/docs/remote/containers
# cat .git/config | grep -A1 "\[user\]" | tail -1 | awk {'print $3'} | xargs --replace=replace git config --global user.email "replace"
# cat .git/config | grep -A1 "\[user\]" | tail -1 | awk {'print $3'} | cut -f1 -d"@" | xargs --replace=replace git config --global user.name "replace"
