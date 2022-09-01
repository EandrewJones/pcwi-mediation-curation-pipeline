#!/usr/bin/env bash

# In order for the script to properly active the virtual environment
# you must add the full path the the venv/bin directory to you PATH
# Environment variable. Please edit the line below to match the path
# to your project directory
PATH=<full-path-to-project-directory>/venv/bin:$PATH

# Set working directory
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path/.."

# Source virtual env
source venv/bin/activate

# Start python job
nohup python3 -u scripts/nu-api-data-fetch-aws-glue.py >> logs/fetch-dispute-data.log &
exit 0
