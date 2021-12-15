#!/bin/bash

# all-in-one fine-tuning script for CV

stage=$1
language=all
mask_name=$3

train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=xlsr_53_56k

homedir=/nobackup/users/clai24/lth-wav2vec/fairseq-march/examples/wav2vec

if [ $stage -eq 4 ]; then
    echo "regular fine-tuning, mimicing LS 1hr fine-tuning setup"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}

    mkdir -p $expdir
    fairseq-hydra-train \
        task.data=$homedir/$datadir \
        model.w2v_path=/nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        checkpoint.save_dir=$expdir \
        --config-dir config/finetuning \
        --config-name xlsr53 \
        2>&1 | tee $expdir/train.log
    exit 0
fi

if [ $stage -eq 100 ]; then
    echo "finetune omp"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot/${mask_name}.pt
    mkdir -p $expdir
fi

if [ $stage -eq 101 ]; then
    echo "finetune omp-cpc"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-cpc/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot-cpc/${mask_name}.pt
    mkdir -p $expdir
fi

if [ $stage -eq 102 ]; then
    echo "finetune rp2"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-rp/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot-rp/${mask_name}.pt
    mkdir -p $expdir
fi

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
    --config-dir config/finetuning \
    --config-name xlsr53-oneshot \
    2>&1 | tee $expdir/train.log

