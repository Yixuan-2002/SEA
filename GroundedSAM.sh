output_file="GroundedSAM_Result.txt"
# Clear the output file if it exists
> $output_file

# FMB Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/GroundedSAM_infer_semseg_MVSeg.py \
    --SAM_ENCODER_VERSION='vit_l' \
    --SAM_CHECKPOINT_PATH="./Grounded-Segment-Anything/sam_vit_l_0b3195.pth" >> $output_file

# MVSeg Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/GroundedSAM_infer_semseg_MVSeg.py \
    --SAM_ENCODER_VERSION='vit_l' \
    --SAM_CHECKPOINT_PATH="./Grounded-Segment-Anything/sam_vit_l_0b3195.pth" >> $output_file