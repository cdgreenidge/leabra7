#!/bin/bash
set -e

echo "Finding package path..."
PACKAGE_PATH=$(conda build scripts/meta.yaml --output)

echo "Uploading package..."
anaconda --token $CONDA_TOKEN upload $PACKAGE_PATH
