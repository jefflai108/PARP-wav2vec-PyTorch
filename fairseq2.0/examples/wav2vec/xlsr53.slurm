#!/bin/bash
#SBATCH -J bash
#SBATCH -o /home/clai24/dump_slurm_logs/xlsr53-prune_%j.out
#SBATCH -e /home/clai24/dump_slurm_logs/xlsr53-prune_%j.err
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
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

stage=$1
lan=$2
rate=$3
chmod -R 777 exp/
chmod u+x ./finetune-commonvoice-xlsr53.sh 
#./finetune-commonvoice-xlsr53.sh 4 $lan
HYDRA_FULL_ERROR=1
./finetune-commonvoice-xlsr53.sh $stage $lan bert_${rate}_mask false
#./finetune-commonvoice-xlsr53.sh 100 $lan bert_${rate}_mask false
#./finetune-commonvoice-xlsr53.sh 101 $lan bert_${rate}_mask false
#./finetune-commonvoice-xlsr53.sh 102 $lan bert_${rate}_mask false

echo "Run completed at:- "
date
