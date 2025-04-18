#!/bin/bash

echo "Running environment setup script..."

# Set path to conda binary
CONDA_BIN=~/miniconda3/bin/conda
PIP_BIN=~/miniconda3/bin/pip

# Check if latentsync environment exists
if $CONDA_BIN env list | grep -q latentsync; then
    echo "Environment latentsync already exists, activating"
else
    echo "Creating conda environment with Python 3.10.13"
    $CONDA_BIN create -y -n latentsync python=3.10.13
fi

# Install ffmpeg
$CONDA_BIN install -y -n latentsync -c conda-forge ffmpeg

# Set path to pip in the latentsync environment
PIP_BIN=~/miniconda3/envs/latentsync/bin/pip

# Python dependencies
echo "Installing Python dependencies..."
$PIP_BIN install -r requirements.txt

# OpenCV dependencies
echo "Installing OpenCV dependencies..."
sudo apt -y install libgl1

# Install huggingface-cli
echo "Installing huggingface-cli..."
$PIP_BIN install huggingface_hub

# Set path to huggingface-cli
HF_CLI=~/miniconda3/envs/latentsync/bin/huggingface-cli

# Download all the checkpoints from HuggingFace
echo "Downloading models from HuggingFace..."
$HF_CLI download ByteDance/LatentSync-1.5 --local-dir checkpoints --exclude "*.git*" "README.md"

# Soft links for the auxiliary models
echo "Creating soft links for models..."
mkdir -p ~/.cache/torch/hub/checkpoints
ln -sf $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
ln -sf $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
ln -sf $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth

echo "Setup is complete!"