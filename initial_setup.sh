#!/bin/bash

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing FFmpeg with NVENC support..."
    sudo apt update
    sudo apt install -y ffmpeg
else
    echo "FFmpeg is already installed"
fi

# Check for NVENC support
if ffmpeg -encoders 2>/dev/null | grep -q nvenc; then
    echo "FFmpeg with NVENC support detected"
else
    echo "NVENC support not detected in FFmpeg"
    echo "Checking NVIDIA drivers..."

    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA drivers are installed"
    else
        echo "NVIDIA drivers not found. Consider installing them for GPU acceleration"
        echo "You can do this with: sudo apt install nvidia-driver-XXX nvidia-cuda-toolkit"
    fi
fi

# Create folders
mkdir -p ~/projects/python
cd ~/projects/python

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/0xthecableguy/LatentSync_x_Blacksmith-lab_version.git
cd LatentSync_x_Blacksmith-lab_version
git fetch --all

# Download and install Miniconda
echo "Installing Miniconda..."
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh

# Copy the fixed setup_env.sh
echo "Preparing setup script..."
cd ~/projects/python/LatentSync_x_Blacksmith-lab_version
cat > tmp_setup_script.sh << 'EOL'
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
EOL

chmod +x tmp_setup_script.sh

# Run the modified setup script
echo "Setting up environment..."
bash tmp_setup_script.sh
rm tmp_setup_script.sh

echo ""
echo "====================================="
echo "Installation complete!"
echo ""
echo "To activate LatentSync environment, run the following commands:"
echo "source ~/miniconda3/etc/profile.d/conda.sh"
echo "conda activate latentsync"
echo "====================================="