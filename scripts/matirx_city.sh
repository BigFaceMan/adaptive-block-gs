set -euo pipefail

DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/street/pose/block_A/"
OUTPUT_PATH="/lfs1/users/spsong/Code/project/adaptive-block-gs/output/mc_block_A_40000"
CUDA_ID=5
ITERATIONS=40000
SWANLAB_EXP_NAME="mc_block_A_40000"

# Check if train and render are already done
if [ -d "$OUTPUT_PATH/point_cloud/iteration_${ITERATIONS}" ] && [ -d "$OUTPUT_PATH/train/ours_${ITERATIONS}/renders" ]; then
    echo "Train and render already completed. Running metrics only..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
else
    echo "Running train and render first..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py \
        -s $DATA_PATH \
        -m $OUTPUT_PATH \
        --iterations $ITERATIONS \
        --test_iterations 7000 $ITERATIONS \
        --save_iterations 7000 $ITERATIONS \
        --checkpoint_iterations 7000 $ITERATIONS \
        --swanlab_experiment_name $SWANLAB_EXP_NAME \
        --eval
    CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py -m $OUTPUT_PATH --iteration $ITERATIONS
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
fi
