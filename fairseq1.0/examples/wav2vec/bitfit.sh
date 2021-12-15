#!/bin/bash 

# finetune only the bias of wav2vec 2.0, based on 
# BitFit: https://nlp.biu.ac.il/~yogo/bitfit.pdf

stage=$1

if [ $stage -eq 2 ]; then 
    # fine-tuning wav2vec_small on 100 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-clean-100
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 1 \
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
        --max-update 80000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --warmup-steps 8000 \
        --hold-steps 32000 \
        --decay-steps 40000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2000000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 8 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 3 ]; then 
    # fine-tuning wav2vec_small on 10 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 301 ]; then 
    echo stage 3 but without freeze-finetune-updates
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 5e-05 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 302 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-04
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 303 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-03
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-03 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 304 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-02
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 305 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-01
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-01 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 306 ]; then 
    echo stage 304 
    echo but with 40k updates
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --max-update 40000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 4000 \
        --hold-steps 16000 \
        --decay-steps 20000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 307 ]; then 
    echo stage 304 
    echo but with 40k updates
    echo but with init learning rate of 2e-02
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --max-update 40000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 2e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 4000 \
        --hold-steps 16000 \
        --decay-steps 20000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 308 ]; then 
    echo stage 304 
    echo but with 40k updates
    echo but with init learning rate of 3e-02
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --max-update 40000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 3e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 4000 \
        --hold-steps 16000 \
        --decay-steps 20000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 309 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-02
    echo with 50k updates 
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
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
        --max-update 50000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 5000 \
        --hold-steps 20000 \
        --decay-steps 25000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 100 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --update-freq 2 \
        --train-with-bitfit \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 3001 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-04
    echo with train-with-bitfit2
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit2 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 3002 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-03
    echo with train-with-bitfit2
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-03 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit2 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 30001 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-04
    echo with train-with-bitfit3
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit3 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 30002 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-03
    echo with train-with-bitfit3
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-03 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 2 \
        --train-with-bitfit3 \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi

if [ $stage -eq 300002 ]; then 
    echo train-with-bitfit4
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --max-update 40000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-03 \
        --lr-scheduler tri_stage \
        --warmup-steps 4000 \
        --hold-steps 16000 \
        --decay-steps 20000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 100 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit4 \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq -300 ]; then 
    echo stage 3 but 
    echo without freeze-finetune-updates
    echo with init learning rate of 1e-02
    echo train with bitfit-random 
    
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
    datadir=data/${train_subset}

    mkdir -p $expdir
    
    python -u src/train_with_bitfit.py \
        --distributed-world-size 2 \
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
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --lr 1e-02 \
        --lr-scheduler tri_stage \
        --warmup-steps 2000 \
        --hold-steps 8000 \
        --decay-steps 10000 \
        --final-lr-scale 0.05 \
        --final-dropout 0.0 \
        --dropout 0.0 \
        --activation-dropout 0.1 \
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 2560000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 100 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --update-freq 4 \
        --train-with-bitfit-random \
        --validate-interval-updates 100 \
        --validate-interval 100 \
        2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 4 ]; then 
    # fine-tuning wav2vec_small on 1 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-1h
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u src/train_with_bitfit.py \
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
        --num-workers 3 \
        --max-update 15000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 3200000 \
        --seed 2337 \
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 1000 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 2 \
        --train-with-bitfit \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        --restore-file $expdir/checkpoint_2502_10000.pt \
        2>&1 | tee $expdir/train.log
fi 

exit 0

if [ $stage -eq 5 ]; then 
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-0
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /data/sls/temp/clai24/lottery-ticket/fairseq/train.py \
        --distributed-world-size 1 \
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
        --max-update 12000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 50 \
        --save-interval 1000 \
        --keep-interval-updates 1 \
        --update-freq 8 2>&1 | tee $expdir/train.log
fi 

if [ $stage -eq 6 ]; then 
    # fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-1
    valid_subset=dev-other
    expdir=exp/${pretrained_model}-finetune-${train_subset}
    datadir=data/${train_subset}

    mkdir -p $expdir

    python -u /data/sls/temp/clai24/lottery-ticket/fairseq/train.py \
        --distributed-world-size 1 \
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
        --max-update 12000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --log-format tqdm \
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

    python -u /data/sls/temp/clai24/lottery-ticket/fairseq/train.py \
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
        --max-update 12000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc \
        --w2v-path /data/sls/temp/clai24/pretrained-models/updated_wav2vecs/${pretrained_model}.pt \
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
        --log-format tqdm \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 50 \
        --save-interval 100 \
        --keep-interval-updates 1 \
        --update-freq 2 2>&1 | tee $expdir/train.log
fi 
