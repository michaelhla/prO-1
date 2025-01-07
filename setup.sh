#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install SWIG
apt-get update
apt-get install swig

# Install Boost development libraries
apt-get install libboost-all-dev

# Install Open Babel
apt install openbabel

# Install requirements
pip install -U numpy vina

# Install requirements
pip install -r requirements.txt

git config --global user.email michaelhla@college.harvard.edu
git config --global user.name michaelhla

echo "set up done, now run 'source venv/bin/activate' to activate the virtual environment"
