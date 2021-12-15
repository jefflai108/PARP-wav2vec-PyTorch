#!/bin/bahs
lan=$1

for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    ./finetune-commonvoice-xlsr53.sh 100 $lan bert_${rate}_mask false
    ./finetune-commonvoice-xlsr53.sh 101 $lan bert_${rate}_mask false
    ./finetune-commonvoice-xlsr53.sh 102 $lan bert_${rate}_mask false
done
