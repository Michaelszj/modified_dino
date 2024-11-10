#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
data_path="./dataset/kinetics_256"

# Search for all subfolders of data_path with depth=1 and extract only the folder names
folder_names=$(find "$data_path" -maxdepth 1 -type d -exec basename {} \; | sort)

# Reverse the folder_names list and convert it to an array
folder_names=($(echo "$folder_names"))

# Delete the first element of the array
unset_indices=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40)

# 使用循环删除指定索引的元素
for index in "${unset_indices[@]}"; do
    unset 'folder_names[index]'
done


for video_id in "${folder_names[@]}"; do
    echo "Processing video $data_path/$video_id"
    python ./preprocessing/save_dino_embed_video.py \
        --config ./config/preprocessing.yaml \
        --data-path $data_path/$video_id

    sleep 2s  # Wait for 2 seconds

    python inference_benchmark.py \
        --config ./config/train.yaml \
        --data-path $data_path/$video_id\
        --benchmark-pickle-path ./tapvid/tapvid_kinetics_data_strided.pkl \
        --video-id $video_id
    rm $data_path/$video_id/dino_embeddings/dino_embed_video.pt
done