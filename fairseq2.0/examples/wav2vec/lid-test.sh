#!/bin/bash

seq_len=$1

language=all
train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=xlsr_53_56k


for batch in 1024; do
    for num_codebooks in 8; do #4 6 8 10
        for codebook_dim in 32; do # 128 64 32
            expdir=exp/${pretrained_model}-commonvoice-lid/$seq_len/dummy
            mkdir -p $expdir
            python -u src/train_lid.py \
                $datadir \
                --task audio_pretraining \
                --path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
                --criterion ctc \
                --log-dir $expdir \
                --seq-len $seq_len \
                --mode train \
                --lan-codebooks $num_codebooks \
                --num-epoches 40 \
                --diversity-alpha 0.1 \
                --lan-codebook-dim $codebook_dim \
                --batch-num $batch \
                --use-spec \
                --inter-lan-entropy \
                --diversity-alpha 0.05 \
                2>&1 | tee $expdir/lid.log
done; done; done
