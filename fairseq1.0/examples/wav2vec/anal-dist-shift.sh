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

if [ $stage -eq 6 ]; then 
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-1
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi 

if [ $stage -eq 7 ]; then 
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-2
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}
fi 

decode_expdir=${expdir}/anal-dist-shift
mkdir -p $decode_expdir
        
python -u src/analyze_distribution_shifts.py \
    $datadir \
    --task audio_pretraining \
    --path $expdir/checkpoint_best.pt \
    --pretrain-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
    --results-path $decode_expdir \
    --outdir $decode_expdir \
    2>&1 | tee $decode_expdir/train.log
