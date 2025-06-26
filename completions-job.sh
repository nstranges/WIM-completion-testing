#!/bin/bash
#SBATCH --account=def-lenck
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-user=nstrang2@uwo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# Wait until DCGM is disabled on the node
while [ ! -z "$(dcgmi -v | grep 'Hostengine build info:')" ]; do
  sleep 5;
done

# Load modules
module load python/3.11
module load gcc arrow/17.0.0
module load cuda
module load mpi4py/4.0.0

# Run on each node
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index datasets
pip install --no-index transformers
pip install --no-index trl

# Activate the environment
source $SLURM_TMPDIR/env/bin/activate;

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch the completions script
python /home/nstrang2/projects/def-lenck/nstrang2/Completions/running-completions.py