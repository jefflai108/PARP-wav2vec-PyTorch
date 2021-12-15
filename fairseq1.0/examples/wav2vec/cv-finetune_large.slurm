#!/bin/bash
#SBATCH -J cv-finetune
#SBATCH -o /home/clai24/dump_slurm_logs/cv_finetune_%j.out
#SBATCH -e /home/clai24/dump_slurm_logs/cv_finetune_%j.err
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/clai24
PYTHON_VIRTUAL_ENVIRONMENT=test2
CONDA_ROOT=$HOME2/tools/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=/home/clai24/dump_slurm_logs/nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

lan=$1

chmod -R 777 exp/
./finetune-commonvoice_large.sh 4 $lan

echo "Run completed at:- "
date
