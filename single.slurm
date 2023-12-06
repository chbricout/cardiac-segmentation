#!/bin/bash
#SBATCH --job-name=brain-vae
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mail-user=bricout.charles@outlook.com
#SBATCH --mail-type=ALL
#SBATCH -o ./logs/output-%x.%j.out # STDOUT

# Load Python module (assuming it's installed on your cluster)
module load python
module load cuda
module load httpproxy
# Create and activate a virtual environment
python -m venv monai
source monai/bin/activate
nvidia-smi nvlink -s
# Install system-level dependencies (adjust as needed)
pip install numpy --no-index
pip install scipy --no-index
pip install scikit-learn --no-index
pip install medpy --no-index
# Install PyTorch and MONAI (adjust versions as needed)
pip install torch torchvision torchaudio --no-index
pip install pandas --no-index
pip install tqdm --no-index

pip install matplotlib --no-index


# Additional dependencies can be installed here if needed
# tensorboard --logdir=tensorboard-runs --host 0.0.0.0 --load_fast false &
python trainer.py
# Deactivate the virtual environment
deactivate