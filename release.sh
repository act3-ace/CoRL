#!/bin/bash -e

# script looks for a variable __version__ in a pyproject.toml file and updates its value of the version
# Matches the examples:
# version = "0.0.0"
# version = '0.0.0'
# use poetry version to update version to next release
NEXT_RELEASE="$1"
poetry version "${NEXT_RELEASE}"
# 
poetry export -f requirements.txt -o requirements.dep.txt --with dev,lint,test,docs,profile
# 
if [ -n "${REGISTRY}" ]; then
    crane auth login -u "${REGISTRY_USER}" -p "${REGISTRY_PASSWORD}" "${REGISTRY}"
    rm -f temp.txt
    for DEST_PATH in $DESTINATION_PATH;
    do
        REPO="${CI_REGISTRY_IMAGE}${DEST_PATH}"
        for tag in $(crane ls "${REPO}")
        do
            echo "$REPO:$tag" >> temp.txt
        done
        echo "${REPO}:v${NEXT_RELEASE}" >> temp.txt

    done
    rm -f registry-images.txt
    grep -E "v[[:digit:]]+.[[:digit:]]+.[[:digit:]]+" temp.txt >> registry-images.txt
    rm -f temp.txt
fi
