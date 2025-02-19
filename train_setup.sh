#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

pip install pyrosetta-installer

# Install requirements, still need to add versions
pip install -r grpo_requirements.txt

pip install unsloth vllm
pip install --upgrade pillow

git config --global user.email michaelhla@college.harvard.edu
git config --global user.name michaelhla

echo "set up done, now run 'source venv/bin/activate' to activate the virtual environment"