#!/bin/bash 

stage=$1
lm=$2
beam=$3
type=$4
mask=$5
ax=$6

if [ "$type" = "regular" ]; then
    echo decoding type: regular
elif [ "$type" = "bitfit" ]; then 
    echo decoding type: bitfit
elif [ "$type" = "oneshot" ]; then 
    echo decoding type: oneshot
else
    echo Example usages are
    echo ./decode.sh 3 false 5 regular 
    echo ./decode.sh 3 true 100 bitfit 
    echo ./decode.sh 304 true 100 bitfit 
    echo ./decode.sh 304 true 100 bitfit none True
    echo ./decode.sh 3 true 100 oneshot bert_0.3_mask
    exit 0
fi

if [ $stage -eq 2 ]; then 
    echo stage 2: fine-tuning wav2vec_small on 100 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-clean-100
    valid_subset=dev-other
    lm_weight=0.87
    word_score=-1
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

if [[ $stage == 3* ]] || [[ $stage == -3* ]]; then
    echo stage 3: fine-tuning wav2vec_small on 10 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10h
    valid_subset=dev-other
    lm_weight=1.06
    word_score=-2.32
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        if [ $stage -eq 3 ]; then 
            expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
        else 
            expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit/stage-${stage}
            [ ! -d $expdir ] && echo $expdir does not exist && exit 0 
        fi 
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

if [ $stage -eq 4 ]; then 
    echo stage 4: fine-tuning wav2vec_small on 1 hr and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-1h
    valid_subset=dev-other
    lm_weight=1.15
    word_score=-2.08
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

if [ $stage -eq 5 ]; then 
    echo stage 5: fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-0
    valid_subset=dev-other
    lm_weight=1.20
    word_score=-1.39
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

if [ $stage -eq 6 ]; then 
    echo stage 6: fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-1
    valid_subset=dev-other
    lm_weight=1.20
    word_score=-1.39
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

if [ $stage -eq 7 ]; then 
    echo stage 7: fine-tuning wav2vec_small on 10 min and validate on dev-other
    pretrained_model=wav2vec_small
    train_subset=train-10min-2
    valid_subset=dev-other
    lm_weight=1.20
    word_score=-1.39
    if [ "$type" = "regular" ]; then
        expdir=exp/${pretrained_model}-finetune-${train_subset}
    elif [ "$type" = "bitfit" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-bitfit
    elif [ "$type" = "oneshot" ]; then 
        expdir=exp/${pretrained_model}-finetune-${train_subset}-oneshot/${mask}
    fi 
fi

datadir=data/${train_subset}

if [ "ax" = true ] ; then
    echo decoding hyper-parm search with Ax

    decode_expdir=${expdir}/ax && mkdir -p $decode_expdir

    for gen_subset in dev-other; do 
            python -u src/infer-ax.py \
                $datadir \
                --task audio_pretraining \
                --nbest 1 \
                --beam ${beam} \
                --seed 1 \
                --batch-size 1 \
                --path $expdir/checkpoint_best.pt \
                --results-path $decode_expdir \
                --w2l-decoder fairseqlm \
                --lm-model /data/sls/temp/clai24/pretrained-models/wav2letter_LMs/lm_librispeech_word_transformer.pt \
                --lexicon /data/sls/temp/clai24/pretrained-models/wav2letter_LMs/librispeech_lexicon.lst \
                --gen-subset $gen_subset \
                --lm-weight 0 \
                --word-score 0 \
                --sil-weight 0 \
                --criterion ctc \
                --labels ltr \
                --max-tokens 4000000 \
                --post-process letter \
                --quiet \
                2>&1 | tee $decode_expdir/decode.log
        done 
    echo finish Ax-search && exit 0
fi 

if [ "$lm" = true ] ; then
    echo "decoding with LM with beam ${beam}"
    
    decode_expdir=${expdir}/decode_translm_nbest1_beam${beam}
    mkdir -p $decode_expdir

    for gen_subset in dev-other dev-clean test-clean test-other; do 
        python -u /data/sls/temp/clai24/lottery-ticket/fairseq/examples/speech_recognition/infer.py \
            $datadir \
            --task audio_pretraining \
            --nbest 1 \
            --beam ${beam} \
            --seed 1 \
            --batch-size 1 \
            --path $expdir/checkpoint_best.pt \
            --results-path $decode_expdir \
            --w2l-decoder fairseqlm \
            --lm-model /data/sls/temp/clai24/pretrained-models/wav2letter_LMs/lm_librispeech_word_transformer.pt \
            --lexicon /data/sls/temp/clai24/pretrained-models/wav2letter_LMs/librispeech_lexicon.lst \
            --gen-subset $gen_subset \
            --lm-weight $lm_weight \
            --word-score $word_score \
            --sil-weight 0 \
            --criterion ctc \
            --labels ltr \
            --max-tokens 2000000 \
            --post-process letter \
            2>&1 | tee $decode_expdir/decode.log
    done 
else
    echo 'decoding without LM'

    decode_expdir=${expdir}/decode 
    mkdir -p $decode_expdir

    for gen_subset in dev-other dev-clean test-clean test-other; do 
        python -u /data/sls/temp/clai24/lottery-ticket/fairseq/examples/speech_recognition/infer.py \
            $datadir \
            --task audio_pretraining \
            --nbest 1 \
            --beam ${beam} \
            --seed 1 \
            --batch-size 40 \
            --path $expdir/checkpoint_best.pt \
            --results-path $decode_expdir \
            --w2l-decoder viterbi \
            --gen-subset $gen_subset \
            --word-score -1 \
            --sil-weight 0 \
            --criterion ctc \
            --labels ltr \
            --max-tokens 4000000 \
            --post-process letter \
            2>&1 | tee $decode_expdir/decode.log
    done 
fi 
