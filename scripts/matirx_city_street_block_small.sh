DATA_PATH="/lfs3/users/spsong/dataset/MatrixCity/small_city/street/pose/block_small/"
OUTPUT_PATH="/lfs1/users/spsong/Code/project/adaptive-block-gs/output/mc_block_small"
CUDA_ID=5
SWANLAB_EXP_NAME="mc_block_A"

# Check if train and render are already done
if [ -d "$OUTPUT_PATH/point_cloud/iteration_30000" ] && [ -d "$OUTPUT_PATH/train/ours_30000/renders" ]; then
    echo "Train and render already completed. Running metrics only..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
else
    echo "Running train and render first..."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py -s $DATA_PATH -m $OUTPUT_PATH --swanlab_experiment_name $SWANLAB_EXP_NAME --eval
    CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py -m $OUTPUT_PATH
    CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
fi