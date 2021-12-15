#!/bin/bash
# OMP on the fine-tuned model

prune_type=$1 # lth/rp/lth_cpc
pretrained_model=xlsr_53_56k
language=$2

expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-2

language=all
expdir=exp/${pretrained_model}-finetune-commonvoice-${language}
datadir=data/commonvoice-${language}-train
echo "expdir is $expdir"
echo "datadir is $datadir"

if [ "$prune_type" = lth ] ; then
    echo "OMP for lth"
    decode_expdir=${expdir}/oneshot
    mkdir -p $decode_expdir

    for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        for comp in bert; do
            python -u src/oneshot.py \
                $datadir \
                --model-type $pretrained_model \
                --task audio_pretraining \
                --path $expdir/checkpoint_best.pt \
                --results-path $decode_expdir \
                --prune-component $comp \
                --rate $rate \
                --outdir $decode_expdir \
                2>&1 | tee $decode_expdir/oneshot.log
        done
    done
elif [ "$prune_type" = rp ]; then
    echo "random pruning"
    decode_expdir=${expdir}/oneshot-rp-2
    mkdir -p $decode_expdir

    for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        for comp in bert; do
            python -u src/oneshot.py \
                $datadir \
                --random-init \
                --omp-cpc \
                --model-type $pretrained_model \
                --task audio_pretraining \
		--path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
                --results-path $decode_expdir \
                --prune-component $comp \
                --rate $rate \
                --outdir $decode_expdir \
                2>&1 | tee $decode_expdir/oneshot.log
        done
    done
elif [ "$prune_type" = lth_cpc ] ; then
    echo "OMP-CPC"
    decode_expdir=${expdir}/oneshot-cpc
    mkdir -p $decode_expdir

    for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        for comp in bert; do
            python -u src/oneshot.py \
                $datadir \
                --omp-cpc \
                --model-type $pretrained_model \
                --task audio_pretraining \
		--path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
                --results-path $decode_expdir \
                --prune-component $comp \
                --rate $rate \
                --outdir $decode_expdir \
                2>&1 | tee $decode_expdir/oneshot.log
        done
    done
else
    echo "$prune_type not supported"; exit 0
fi
