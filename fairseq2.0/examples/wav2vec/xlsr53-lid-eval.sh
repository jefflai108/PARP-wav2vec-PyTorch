#!/bin/bash

seq_len=$1

language=all
train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=xlsr_53_56k


for batch in 256; do
    for num_codebooks in 4 6 8 10; do #4 6 8 10
        for codebook_dim in 128 64 32; do # 128 64 32
            expdir=exp/${pretrained_model}-commonvoice-lid/$seq_len/batch${batch}_codebook-num${num_codebooks}_codebook-dim${codebook_dim}
            decodedir=$expdir/decode
            mkdir -p $decodedir
            python -u src/train_lid.py \
                $datadir \
                --task audio_pretraining \
                --path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
                --criterion ctc \
                --log-dir $expdir \
                --seq-len $seq_len \
                --mode eval \
                --lan-codebooks $num_codebooks \
                --num-epoches 40 \
                --diversity-alpha 0.1 \
                --lan-codebook-dim $codebook_dim \
                --use-spec \
                2>&1 | tee $decodedir/decode.log
done; done; done
