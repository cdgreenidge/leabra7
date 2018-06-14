#!/bin/bash
set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ]
then
    echo "Not on master branch, canceling deployment to Anaconda Cloud."
    exit 0
elif [ "${TRAVIS_PULL_REQUEST}" = "true" ]
then
    echo "This is a pull request, canceling deployment to Anaconda Cloud."
    exit 0
else
    echo "Deploying to Anaconda Cloud..."
fi

echo "Finding package path..."
PACKAGE_PATH=$(conda build scripts/meta.yaml --output)

echo "Uploading package..."
anaconda --token $CONDA_TOKEN upload $PACKAGE_PATH
