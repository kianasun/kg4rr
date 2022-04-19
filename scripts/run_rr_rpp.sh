#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

MODEL=$1
DIM=$2

echo "Start Evaluating......"

python src/main/run_evaluate.py \
    --data_path dataset/rpp/ \
    --checkpoint dataset/rpp/models/$MODEL"_"$DIM/checkpoint.pt\
    --content dataset/rpp/content_emb.npy \
    --numeric dataset/rpp/num_feat.npy \
    --valid_node node2lab.csv  \
    --model $MODEL\
    --dim $DIM \
    --device cpu \
    $3 $4
