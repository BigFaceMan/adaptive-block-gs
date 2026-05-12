DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/street/pose/block_1"
OUTPUT_PATH="/lfs1/users/spsong/Code/gaussian-splatting/output/single_image_test"
CUDA_ID=5
SWANLAB_EXP_NAME="single_image_test"

CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py \
    -s $DATA_PATH \
    -m $OUTPUT_PATH \
    --swanlab_experiment_name $SWANLAB_EXP_NAME \
    --iterations 1000 \
    --save_iterations 500 1000 \
    --test_iterations 500 1000
