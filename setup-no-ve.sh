# This shell is to setup the environment on cloud platform WITHOUT virtual enviroment like conda.
# Make sure python = 3.12

# Install dependencies
pip install git+https://github.com/flashinfer-ai/flashinfer-bench.git@main modal

# Download dataset.
# Install git-lfs first.
apt-get update && apt-get install -y git-lfs
# Download dataset.
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH="/root/flashinfer/mlsys26-contest"

# Pack solution
python scripts/pack_solution.py

# Run benchmark locally.
python scripts/run_local.py