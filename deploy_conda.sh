#!/bin/bash
set -e

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ $BRANCH = "master" ]
then
    echo "Deploying to Anaconda Cloud..."
else
    echo "Not on master branch, canceling deployment to Anaconda Cloud."
    exit 0
fi

echo "Finding package path..."
PACKAGE_PATH=$(conda build . --output)

echo "Uploading package..."
anaconda --token $CONDA_TOKEN upload $PACKAGE_PATH
