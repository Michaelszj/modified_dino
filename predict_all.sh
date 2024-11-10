for video_id in {0..29}; do
    python inference_benchmark.py \
    --config ./config/train.yaml \
    --data-path ./dataset/davis_256/$video_id\
    --benchmark-pickle-path ./tapvid/tapvid_davis_data_strided.pkl \
    --video-id $video_id
done