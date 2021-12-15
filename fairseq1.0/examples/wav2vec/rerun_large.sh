#!/bin/bash

stage=$1

if [ $stage -eq 2 ]; then
    # fine-tuning libri960_big on 100 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-clean-100
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --best-checkpoint-metric wer \
        --num-workers 8 \
        --max-update 50000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 3e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 5000 \
        --hold-steps 20000 \
        --decay-steps 25000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 1280000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 5 \
        --save-interval 5 \
        --keep-interval-updates 1 \
        --validate-interval-updates 5 \
        --validate-interval 5 \
        --update-freq 5 \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 3 ]; then
    # fine-tuning libri960_big on 100 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-rerun
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 8 \
        --max-update 20000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 1e-04 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 1280000 \
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
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 4 ]; then
    # fine-tuning wav2vec_small on 1 hr and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-1h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-rerun
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 4 \
        --max-update 15000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 1500 \
        --hold-steps 6000 \
        --decay-steps 7500 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 1280000 \
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
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 5 ]; then
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=libri960_big
    train_subset=train-10min-0
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-rerun
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 4 \
        --max-update 12000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 1200 \
        --hold-steps 4800 \
        --decay-steps 6000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.1 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 1280000 \
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
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 6 ]; then
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-1
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 1 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --best-checkpoint-metric wer \
        --num-workers 8 \
        --max-update 12000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 1200 \
        --hold-steps 4800 \
        --decay-steps 6000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 3200000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 50 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 8 2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 7 ]; then
	## (4-gpu) fine-tuning
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-2
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --fp16 \
        --post-process letter \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --best-checkpoint-metric wer \
        --num-workers 8 \
        --max-update 12000 \
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
        --freeze-finetune-updates 10000 \
        --validate-after-updates 10000 \
        --optimizer adam \
        --adam-betas '(0.9, 0.98)' \
        --adam-eps 1e-08 \
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 1200 \
        --hold-steps 4800 \
        --decay-steps 6000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 3200000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 50 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --update-freq 2 2>&1 | tee $expdir/train.log
fi

exit 0

if [ $stage -eq 300 ]; then
	## train from scratch commands
    train_subset=train-960
    valid_subset=dev-clean
    expdir=exp/${train_subset}/distill_ce-att_head_6-embed_dim_480-ffn_embed_dim_1920-layer_6-latent_vars_160
    datadir=data/${train_subset}
    mkdir -p $expdir


        echo 'evaluating'

        finetune_train_subset=train-clean-100
        finetune_valid_subset=dev-clean
        finetune_datadir=data/${finetune_train_subset}
        finetune_expdir=${expdir}/fine_tune-${finetune_train_subset}
        mkdir -p $finetune_expdir/decode

        python /nobackup/users/clai24/knowledge-transfer/fairseq/examples/speech_recognition/infer.py data/train-clean-100/ --task audio_pretraining \
            --nbest 1 --path $finetune_expdir/checkpoint_best.pt --results-path $finetune_expdir/decode --w2l-decoder viterbi \
            --gen-subset dev-clean \
            --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
            --post-process letter

        ## train from scratch commands
        echo 'pretraining'

        python /nobackup/users/clai24/knowledge-transfer/fairseq/train.py --distributed-world-size ${ngpu} --distributed-port 0 $datadir --save-dir $expdir \
            --train-subset $train_subset --valid-subset $valid_subset \
            --num-workers 8 --task audio_pretraining --criterion wav2vec_kd --arch wav2vec2 \
            --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
            --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 256 \
            --latent-groups 2 --latent-temp '(2,0.5,0.999995)' --infonce --optimizer adam \
            --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 400000 \
            --lr 0.0005 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
            --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 \
            --loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
            --max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
            --max-tokens 1400000 --max-update 400000 --skip-invalid-size-inputs-valid-test --log-format json --ddp-backend no_c10d \
            --encoder-attention-heads 6 --encoder-embed-dim 480 --encoder-ffn-embed-dim 1920 --encoder-layers 6 --latent-vars 160 \
            --distill-with-ce 2>&1 | tee $expdir/train.log
fi

