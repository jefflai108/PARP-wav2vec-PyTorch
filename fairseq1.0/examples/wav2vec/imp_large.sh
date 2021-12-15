#!/bin/bash

stage=$1
rewind_init=$2
target_prune_rate=$3

if [ $stage -eq 2 ]; then
    # fine-tuning libri960_big on 100 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-clean-100
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u src/train_with_imp.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --best-checkpoint-metric wer \
        --num-workers 8 \
        --max-update 5000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        --labels ltr \
        --apply-mask \
        --mask-selection static \
        --mask-other 0 \
        --mask-length 10 \
        --mask-prob 0.5 \
        --mask-channel-selection static \
        --mask-channel-other 0 \
        --mask-channel-length 64 \
        --mask-channel-prob 0.512 \
        --zero-infinity \
        --feature-grad-mult 0.0 \
        --freeze-finetune-updates 0 \
        --validate-after-updates 0 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 3e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 500 \
        --hold-steps 2000 \
        --decay-steps 2500 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 600000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 200 \
        --save-interval 200  \
        --keep-interval-updates 1 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        --update-freq 5 \
        --restore-file $rewind_init_ckpt \
        --target-prune-rate $target_prune_rate \
        --reset-optimizer \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 3 ]; then
    echo "IMP fine-tuning on 10hr starting from $rewind_init with target prune rate of $target_prune_rate"
    pretrained_model=libri960_big
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}/imp/${rewind_init}/bert_${target_prune_rate}_mask
    datadir=data/${train_subset}
    rewind_init_ckpt=exp/${pretrained_model}-finetune-${train_subset}/${rewind_init}.pt

    [ ! -f $rewind_init_ckpt ] && echo "$rewind_init_ckpt does not exist" && exit 0

    mkdir -p $expdir

    python -u src/train_with_imp.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 4 \
        --max-update 2000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        --labels ltr \
        --apply-mask \
        --mask-selection static \
        --mask-other 0 \
        --mask-length 10 \
        --mask-prob 0.65 \
        --mask-channel-selection static \
        --mask-channel-other 0 \
        --mask-channel-length 64 \
        --mask-channel-prob 0.256 \
        --zero-infinity \
        --feature-grad-mult 0.0 \
        --freeze-finetune-updates 0 \
        --validate-after-updates 0 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 1e-04 \
        --lr-scheduler tri_stage \
        --warmup-steps 200 \
        --hold-steps 800 \
        --decay-steps 1000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 600000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 20 \
        --save-interval 20 \
        --keep-interval-updates 1 \
        --validate-interval-updates 20 \
        --validate-interval 20 \
        --update-freq 5 \
        --restore-file $rewind_init_ckpt \
        --target-prune-rate $target_prune_rate \
        --reset-optimizer \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 4 ]; then
    echo "IMP fine-tuning on 1hr starting from $rewind_init with target prune rate of $target_prune_rate"
    pretrained_model=libri960_big
    train_subset=train-1h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}/imp/${rewind_init}/bert_${target_prune_rate}_mask
    datadir=data/${train_subset}
    rewind_init_ckpt=exp/${pretrained_model}-finetune-${train_subset}/${rewind_init}.pt

    [ ! -f $rewind_init_ckpt ] && echo "$rewind_init_ckpt does not exist" && exit 0

    mkdir -p $expdir

    python -u src/train_with_imp.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 4 \
        --max-update 2000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        --labels ltr \
        --apply-mask \
        --mask-selection static \
        --mask-other 0 \
        --mask-length 10 \
        --mask-prob 0.75 \
        --mask-channel-selection static \
        --mask-channel-other 0 \
        --mask-channel-length 64 \
        --mask-channel-prob 0.256 \
        --zero-infinity \
        --feature-grad-mult 0.0 \
        --freeze-finetune-updates 0 \
        --validate-after-updates 0 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 200 \
        --hold-steps 800 \
        --decay-steps 1000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 600000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 100 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        --update-freq 5 \
        --restore-file $rewind_init_ckpt \
        --target-prune-rate $target_prune_rate \
        --reset-optimizer \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 5 ]; then
    echo "IMP fine-tuning on 10min starting from $rewind_init with target prune rate of $target_prune_rate"
    pretrained_model=libri960_big
    train_subset=train-10min-0
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}/imp/${rewind_init}/bert_${target_prune_rate}_mask
    datadir=data/${train_subset}
    rewind_init_ckpt=exp/${pretrained_model}-finetune-${train_subset}/${rewind_init}.pt

    [ ! -f $rewind_init_ckpt ] && echo "$rewind_init_ckpt does not exist" && exit 0

    mkdir -p $expdir

    python -u src/train_with_imp.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 4 \
        --max-update 2000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /nobackup/users/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
        --labels ltr \
        --apply-mask \
        --mask-selection static \
        --mask-other 0 \
        --mask-length 10 \
        --mask-prob 0.75 \
        --mask-channel-selection static \
        --mask-channel-other 0 \
        --mask-channel-length 64 \
        --mask-channel-prob 0.512 \
        --zero-infinity \
        --feature-grad-mult 0.0 \
        --freeze-finetune-updates 0 \
        --validate-after-updates 0 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 200 \
        --hold-steps 800 \
        --decay-steps 1000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 600000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 200 \
        --save-interval 200 \
        --keep-interval-updates 1 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        --update-freq 5 \
        --restore-file $rewind_init_ckpt \
        --target-prune-rate $target_prune_rate \
        --reset-optimizer \
        2>&1 | tee $expdir/train.log
fi

