#!/bin/bash

language=$1
mask_name=$2
cross_lin=${3:-false}
cross_lin_lan=$4

train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=xlsr_53_56k

homedir=/nobackup/users/clai24/lth-wav2vec/fairseq-march/examples/wav2vec

echo "omp-cpc mask + make pruning permanent + prune at the end"
expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-2-oneshot-permanent-freq10/${mask_name}
mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}-2/oneshot-cpc/${mask_name}.pt
if [ "$cross_lin" = true ]; then
    echo "cross-lingual masking:"
    echo "mask from ${cross_lin_lan}"
    echo "fine-tune on ${language}"
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${cross_lin_lan}-2/oneshot/${mask_name}.pt
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-2-oneshot-permanent/${cross_lin_lan}-${mask_name}
fi

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
    task.epoch_interval_for_pruning=10 \
    --config-dir config/finetuning \
    --config-name xlsr53-2-oneshot-permanent2 \
    2>&1 | tee $expdir/train.log

