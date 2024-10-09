output_file="xdecoder_Result.txt"
# Clear the output file if it exists
> $output_file

# FMB Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/xdecoder_infer_semseg_FMB.py evaluate \
    --conf_files configs/seem/focall_unicl_lang_v1.yaml --overrides RESUME_FROM checkpoints/seem_focall_v1.pt >> $output_file

# MVSeg Dataset
CUDA_VISIBLE_DEVICES=1 python -W ignore inference/xdecoder_infer_semseg_MVSeg.py evaluate \
    --conf_files configs/xdecoder/focall_unicl_lang.yaml --overrides RESUME_FROM checkpoints/xdecoder_focall_last.pt >> $output_file