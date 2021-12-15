#!/bin/bash
# OMP on the fine-tuned model

stage=$1

if [ $stage -eq 2 ]; then
    # fine-tuning wav2vec_small on 100 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-clean-100
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 3 ]; then
    # fine-tuning wav2vec_small on 10 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 4 ]; then
    # fine-tuning wav2vec_small on 1 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-1h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 5 ]; then
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-0
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

#### large model below ####

if [ $stage -eq 8 ]; then
    # fine-tuning libri960_big on 100 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-clean-100
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 9 ]; then
    # fine-tuning libri960_big on 10 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 10 ]; then
    # fine-tuning libri960_big on 1 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-1h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

if [ $stage -eq 11 ]; then
    # fine-tuning libri960_big on 10 min and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-10min-0
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi

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
