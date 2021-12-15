#!/bin/bash

mask_name=$1

train_subset=train-1h
valid_subset=dev-other
datadir=data/${train_subset}
pretrained_model=xlsr_53_56k

homedir=/nobackup/users/clai24/lth-wav2vec/fairseq-march/examples/wav2vec

echo "omp-cpc mask + make pruning permanent + prune at the end"
expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot-permanent/${mask_name}
mask_file=exp/xlsr_53_56k-finetune-commonvoice-all/oneshot-cpc/${mask_name}.pt

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
    --config-dir config/finetuning \
    --config-name vox_1h \
    2>&1 | tee $expdir/train.log

