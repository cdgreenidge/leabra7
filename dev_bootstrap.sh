#!/bin/bash

# This script will set you up for developing the leabra7 project, from scratch.
# Run it with "source dev_bootstrap.sh"
PROJECT=leabra7

# If necessary, change these variables to match your python installation
PYTHON=python3
PIP=pip3
VIRTUALENV_DIR=$HOME/.virtualenvs/

# Create and activate a new virtual environment
$PYTHON -m venv $VIRTUALENV_DIR$PROJECT
source $VIRTUALENV_DIR/$PROJECT/bin/activate

# Install the project in development mode, along with the dev dependencies
$PIP install -e ".[dev]"
