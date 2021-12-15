#!/bin/bash

stage=$1

pretrained_model=wav2vec_vox_new
homedir=/nobackup/users/clai24/lth-wav2vec/fairseq-march/examples/wav2vec

if [ $stage -eq 3 ]; then
train_subset=train-10min-0
valid_subset=dev-other
datadir=data/${train_subset}

expdir=exp/${pretrained_model}-finetune-${train_subset}
mkdir -p $expdir

echo "expdir: $expdir"

    HYDRA_FULL_ERROR=1
    fairseq-hydra-train \
        task.data=$homedir/$datadir \
        model.w2v_path=/nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        checkpoint.save_dir=$expdir \
        --config-dir config/finetuning \
        --config-name vox_10m \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 4 ]; then
train_subset=train-10min-0
valid_subset=dev-other
datadir=data/${train_subset}

expdir=exp/${pretrained_model}-finetune-${train_subset}
mkdir -p $expdir

echo "expdir: $expdir"

    HYDRA_FULL_ERROR=1
    fairseq-hydra-train \
        task.data=$homedir/$datadir \
        model.w2v_path=/nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        checkpoint.save_dir=$expdir \
        --config-dir config/finetuning \
        --config-name vox_10m \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 5 ]; then
train_subset=train-10min-0
valid_subset=dev-other
datadir=data/${train_subset}

expdir=exp/${pretrained_model}-finetune-${train_subset}
mkdir -p $expdir

echo "expdir: $expdir"

    HYDRA_FULL_ERROR=1
    fairseq-hydra-train \
        task.data=$homedir/$datadir \
        model.w2v_path=/nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        dataset.train_subset=$train_subset \
        dataset.valid_subset=$valid_subset \
        checkpoint.save_dir=$expdir \
        --config-dir config/finetuning \
        --config-name vox_10m \
        2>&1 | tee $expdir/train.log
fi
