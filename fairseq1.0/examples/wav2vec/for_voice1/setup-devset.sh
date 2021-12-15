#!/bin/bash 

stage=$1

if [ $stage -eq 1 ]; then 
    echo "set up dev set for the training set"
    for source in dev-other dev-clean test-clean test-other; do 
    for target in train-clean-100 train-clean-360 train-960 train-1h train-10h train-10min-0 train-10min-1 train-10min-2 train-10min-3 train-10min-4 train-10min-5; do
        cp data/$source/${source}.ltr data/$target/
        cp data/$source/${source}.wrd data/$target/ 
        cp data/$source/train.tsv data/$target/${source}.tsv 
        cp data/$target/train.tsv data/$target/${target}.tsv 
        cp dict.ltr.txt data/$target/ 
    done; done 
fi 

if [ $stage -eq 2 ]; then 
    echo "set up small dev set for Ax optimization"
    for source in dev-other; do 
    for target in train-clean-100 train-clean-360 train-960 train-1h train-10h train-10min-0 train-10min-1 train-10min-2 train-10min-3 train-10min-4 train-10min-5; do
        head -200 data/$source/${source}.ltr > data/$target/${source}-200.ltr
        head -200 data/$source/${source}.wrd > data/$target/${source}-200.wrd 
        head -201 data/$source/train.tsv > data/$target/${source}-200.tsv
    done; done 
fi 

if [ $stage -eq 3 ]; then 
    echo "set up dev set for CommonVoice"
    for lan in es fr it ky nl ru sv_SE tr tt zh_TW; do 
        cp data/commonvoice-${lan}-val/${lan}-val.ltr data/commonvoice-${lan}-train/
        cp data/commonvoice-${lan}-test/${lan}-test.ltr data/commonvoice-${lan}-train/
        cp data/commonvoice-${lan}-val/${lan}-val.wrd data/commonvoice-${lan}-train/
        cp data/commonvoice-${lan}-test/${lan}-test.wrd data/commonvoice-${lan}-train/
        cp data/commonvoice-${lan}-val/train.tsv data/commonvoice-${lan}-train/${lan}-val.tsv
        cp data/commonvoice-${lan}-test/train.tsv data/commonvoice-${lan}-train/${lan}-test.tsv
        cp data/commonvoice-${lan}-train/train.tsv data/commonvoice-${lan}-train/${lan}-train.tsv
        #cp /data/sls/d/corpora/processed/commonvoice/data_riviere2020/kaldi_style/${lan}_lang_char/train_${lan}_1hr_units.txt data/commonvoice-${lan}-train/dict.ltr.txt
    done
fi 

