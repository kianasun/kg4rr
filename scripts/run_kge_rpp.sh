#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

GPU_DEVICE=$1
MODEL=$2
DIM=$3
EPOCH=$4

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/main/run_kge.py \
    --triplets dataset/rpp/triples.txt \
    --id2ent dataset/rpp/entities.dict \
    --id2rel dataset/rpp/relations.dict \
    --save dataset/rpp/models/$MODEL"_"$DIM/ \
    --model $MODEL --dim $DIM --epoch $EPOCH \
    --content dataset/rpp/content_emb.npy \
    --numeric dataset/rpp/num_feat.npy

