# if [ -f "$HOME/.kube-ps1/kube-ps1.sh" ]; then . "$HOME/.kube-ps1/kube-ps1.sh"; fi

if [ -f "$HOME/.bash-git-prompt/gitprompt.sh" ]; then
    GIT_PROMPT_ONLY_IN_REPO=1
    GIT_PROMPT_END='\n\$ '
    source $HOME/.bash-git-prompt/gitprompt.sh
fi
