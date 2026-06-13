#!/bin/bash
#SBATCH --account=def-thocking_cpu
#SBATCH --job-name=collect-env
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=env-%j.txt

echo "### NODE ###"; hostname; date
echo "### OS ###"; cat /etc/os-release
echo "### CPU ###"; lscpu
echo "### MEMORY (node total) ###"; free -h
echo "### THIS JOB ALLOCATION ###"
echo "node=$SLURM_JOB_NODELIST cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}MB"
echo "### PYTHON ###"; module load python/3.11.5; python --version