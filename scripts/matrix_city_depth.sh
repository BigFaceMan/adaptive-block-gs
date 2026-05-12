set -euo pipefail

DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/street/pose/block_A"
DEPTH_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city_depth/street"
OUTPUT_PATH="/lfs1/users/spsong/Code/gaussian-splatting/output/mc_block_A_depth"
CUDA_ID=5
SWANLAB_EXP_NAME="mc_block_A_depth"

export OPENCV_IO_ENABLE_OPENEXR=1

if [ -d "$OUTPUT_PATH/point_cloud/iteration_30000" ] && [ -d "$OUTPUT_PATH/test/ours_30000/renders" ]; then
    echo "Train and render already completed. Running metrics only..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
else
    echo "Running train, render, and metrics with depth regularization..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py \
        -s $DATA_PATH \
        -d $DEPTH_PATH \
        -m $OUTPUT_PATH \
        --swanlab_experiment_name $SWANLAB_EXP_NAME \
        --checkpoint_iterations 7000 30000 \
        --eval
    CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py -m $OUTPUT_PATH --skip_train
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
fi
