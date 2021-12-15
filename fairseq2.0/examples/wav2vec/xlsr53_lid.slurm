#!/bin/bash
#SBATCH -J xlsr53
#SBATCH -o /home/clai24/dump_slurm_logs/xlsr53_%j.out
#SBATCH -e /home/clai24/dump_slurm_logs/xlsr53_%j.err
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/clai24
PYTHON_VIRTUAL_ENVIRONMENT=fairseq2
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

seq_len=$1
chmod -R 777 exp/
chmod u+x ./xlsr53-lid.sh
./xlsr53-lid.sh $seq_len

echo "Run completed at:- "
date
