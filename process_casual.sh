target=$1


python ./preprocessing/main_preprocessing.py \
    --config ./config/preprocessing.yaml \
    --data-path ./casual_video/$target 

python ./train.py \
    --config ./config/train.yaml \
    --data-path ./casual_video/$target 

python ./inference_grid.py \
    --config ./config/train.yaml \
    --data-path ./casual_video/$target 