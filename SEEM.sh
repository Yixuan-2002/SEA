output_file="SEEM_Result.txt"
# Clear the output file if it exists
> $output_file

# FMB Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/SEEM_infer_semseg_FMB.py >> $output_file

# MVSeg Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/SEEM_infer_semseg_MVSeg.py >> $output_file