#!/bin/bash

language=$1
init_mask=$2
final_mask=$3
target_prune_rate=$4

train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=xlsr_53_56k

homedir=/nobackup/users/clai24/lth-wav2vec/fairseq-march/examples/wav2vec

echo "omp-cpc mask + make pruning permanent + progressive prune at the end"
expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-2-oneshot-permanent-growing/${final_mask}
mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}-2/oneshot-cpc/${init_mask}.pt

mkdir -p $expdir
echo "mask file from $mask_file"
echo "expdir: $expdir"

HYDRA_FULL_ERROR=1
fairseq-hydra-train \
    task.data=$homedir/$datadir \
    model.w2v_path=/nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
    dataset.train_subset=$train_subset \
    dataset.valid_subset=$valid_subset \
    checkpoint.save_dir=$expdir \
    task.mask_file=$homedir/$mask_file \
    task.epoch_interval_for_pruning=5 \
    task.target_prune_rate=$target_prune_rate \
    task.total_epochs=3000 \
    --config-dir config/finetuning \
    --config-name xlsr53-2-oneshot-permanent-growing4 \
    2>&1 | tee $expdir/train.log

