#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install system packages with automatic yes
sudo yum update -y
sudo yum install -y swig

# Install Boost development libraries with automatic yes
sudo yum install -y boost-devel

# Install Open Babel with automatic yes
sudo yum install -y openbabel

pip install -U numpy vina
pip install pyrosetta-installer

pip install unsloth
pip install --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# Install requirements, still need to add versions
pip install -r requirements.txt

git config --global user.email michaelhla@college.harvard.edu
git config --global user.name michaelhla

echo "set up done, now run 'source venv/bin/activate' to activate the virtual environment"
