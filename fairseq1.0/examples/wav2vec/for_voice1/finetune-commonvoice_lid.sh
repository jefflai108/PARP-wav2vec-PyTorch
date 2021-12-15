#!/bin/bash

# all-in-one fine-tuning script for CV

stage=$1
language=$2
mask_name=$3
cross_lin=${4:-false}
cross_lin_lan=$5

if [ $stage -eq 0 ]; then
    for lan in es fr it ky nl ru sv_SE tr tt zh_TW; do
        for split in train val test; do
            mkdir -p data/commonvoice-${lan}-${split}
            python -u wav2vec_manifest.py /home/clai24/common_voices_splits/${lan}/clips_16k-${split} \
                                            --dest data/commonvoice-${lan}-${split} --ext flac --valid-percent 0
        done
    done
fi

if [ $stage -eq 1 ]; then
    for lan in es fr it ky nl ru sv_SE tr tt zh_TW; do
        for split in train val test; do
            python libri_labels.py data/commonvoice-${lan}-${split}/train.tsv \
                                    --output-dir data/commonvoice-${lan}-${split} \
                                    --output-name ${lan}-${split} \
                                    --transcription /home/clai24/common_voices_splits/${lan}/${split}_utt2transcript-phn
            cp data/commonvoice-${lan}-${split}/${lan}-${split}.wrd data/commonvoice-${lan}-${split}/${lan}-${split}.ltr # only for phn prediction
            cp /home/clai24/common_voices_splits/${lan}/dict.ltr.txt data/commonvoice-${lan}-${split} # copy dict.ltr.txt
        done
    done
    ./setup-devset.sh 3
fi

train_subset=${language}-train
valid_subset=${language}-val
datadir=data/commonvoice-${language}-train
pretrained_model=wav2vec_small

if [ $stage -eq 4 ]; then
    echo "regular fine-tuning, mimicing LS 1hr fine-tuning setup"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-LID

    mkdir -p $expdir

    python -u /nobackup/users/clai24/lth-wav2vec/lth-wav2vec/train.py \
        --distributed-world-size 4 \
        --distributed-port 0 \
        $datadir \
        --save-dir $expdir \
        --train-subset $train_subset \
        --valid-subset $valid_subset \
        --no-epoch-checkpoints \
        --best-checkpoint-metric wer \
        --num-workers 0 \
        --max-update 15000 \
        --sentence-avg \
        --task audio_pretraining \
        --arch wav2vec_ctc_oneshot \
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
        --layerdrop 0.05 \
        --criterion ctc \
        --attention-dropout 0.0 \
        --max-tokens 3200000 \
        --seed 2337 \
        --log-format json \
        --log-interval=200 \
        --ddp-backend no_c10d \
        --save-interval-updates 200 \
        --save-interval 200 \
        --keep-interval-updates 1 \
        --update-freq 2 \
        --validate-interval-updates 200 \
        --validate-interval 200 \
        2>&1 | tee $expdir/train.log
    exit 0
fi

if [ $stage -eq 100 ]; then
    echo "finetune omp"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot/${mask_name}.pt
    if [ "$cross_lin" = true ]; then
        echo "cross-lingual masking:"
        echo "mask from ${cross_lin_lan}"
        echo "fine-tune on ${language}"
        mask_file=exp/${pretrained_model}-finetune-commonvoice-${cross_lin_lan}/oneshot/${mask_name}.pt
        expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot/${cross_lin_lan}-${mask_name}
    fi

    mkdir -p $expdir
fi

if [ $stage -eq 101 ]; then
    echo "finetune omp-cpc"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-cpc/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot-cpc/${mask_name}.pt
    if [ "$cross_lin" = true ]; then
        echo "cross-lingual masking:"
        echo "mask from ${cross_lin_lan}"
        echo "fine-tune on ${language}"
        mask_file=exp/${pretrained_model}-finetune-commonvoice-${cross_lin_lan}/oneshot-cpc/${mask_name}.pt
        expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-cpc/${cross_lin_lan}-${mask_name}
    fi

    mkdir -p $expdir
fi

if [ $stage -eq 102 ]; then
    echo "finetune rp2"
    expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-rp-2/${mask_name}
    mask_file=exp/${pretrained_model}-finetune-commonvoice-${language}/oneshot-rp-2/${mask_name}.pt
    if [ "$cross_lin" = true ]; then
        echo "cross-lingual masking:"
        echo "mask from ${cross_lin_lan}"
        echo "fine-tune on ${language}"
        mask_file=exp/${pretrained_model}-finetune-commonvoice-${cross_lin_lan}/oneshot-rp-2/${mask_name}.pt
        expdir=exp/${pretrained_model}-finetune-commonvoice-${language}-oneshot-rp-2/${cross_lin_lan}-${mask_name}
    fi

    mkdir -p $expdir
fi

echo "mask file from $mask_file"
echo "expdir: $expdir"

python -u src/train_with_oneshot.py \
    --distributed-world-size 4 \
    --distributed-port 0 \
    $datadir \
    --save-dir $expdir \
    --train-subset $train_subset \
    --valid-subset $valid_subset \
    --no-epoch-checkpoints \
    --best-checkpoint-metric wer \
    --num-workers 0 \
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
    --layerdrop 0.05 \
    --criterion ctc \
    --attention-dropout 0.0 \
    --max-tokens 3200000 \
    --seed 2337 \
    --log-format json \
    --log-interval=200 \
    --ddp-backend no_c10d \
    --save-interval-updates 200 \
    --save-interval 200 \
    --keep-interval-updates 1 \
    --update-freq 2 \
    --validate-interval-updates 200 \
    --validate-interval 200 \
    --mask-file $mask_file \
    2>&1 | tee $expdir/train.log

