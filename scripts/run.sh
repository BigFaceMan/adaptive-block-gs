###
 # @Author: ssp
 # @Date: 2025-02-10 21:02:01
 # @LastEditTime: 2025-11-25 00:16:35
### 
DATA_PATH="/lfs3/users/spsong/Waymo/waymo2colmap/waymo_01"

OUTPUT_PATH="/lfs1/users/spsong/Code/project/adaptive-block-gs/output/waymo_01"
CUDA_ID=1
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py -s $DATA_PATH -m $OUTPUT_PATH
CUDA_VISIBLE_DEVICES=$CUDA_ID python render.py -m $OUTPUT_PATH
CUDA_VISIBLE_DEVICES=$CUDA_ID python metrics.py -m $OUTPUT_PATH
