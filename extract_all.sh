#!/bin/bash


for video_id in {0..29}; do
    python ./preprocessing/save_dino_embed_video.py \
        --config ./config/preprocessing.yaml \
        --data-path ./dataset/davis_256/$video_id

    python inference_benchmark.py \
    --config ./config/train.yaml \
    --data-path ./dataset/davis_256/$video_id\
    --benchmark-pickle-path ./tapvid/tapvid_davis_data_strided.pkl \
    --video-id $video_id

    rm -rf ./dataset/davis_256/$video_id/dino_embeddings/dino_embed_video.pt
done