#!/bin/bash

for video_id in {0..29}; do
    rm ./dataset/davis_256/$video_id/dino_embeddings/refined_embed_video.pt
done