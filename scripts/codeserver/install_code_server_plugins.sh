#!/usr/bin/env bash

echo "Install the VSCODE plugins"
while read p; do
  echo "$p"
  /usr/bin/code-server --install-extension ${p}
done <scripts/codeserver/default_code_server_plugins.txt
