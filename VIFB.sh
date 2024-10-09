#!/bin/bash

output_file="VIFB_metrics_output.txt"

# Clear the output file if it exists
> $output_file

python3 VIFB/a_VIFB_metrics_AG.py >> $output_file
python3 VIFB/a_VIFB_metrics_CE.py >> $output_file
python3 VIFB/a_VIFB_metrics_EN.py >> $output_file
python3 VIFB/a_VIFB_metrics_MI.py >> $output_file
python3 VIFB/a_VIFB_metrics_PSNR.py >> $output_file
python3 VIFB/a_VIFB_metrics_Qabf.py >> $output_file
python3 VIFB/a_VIFB_metrics_CC.py >> $output_file
python3 VIFB/a_VIFB_metrics_QC.py >> $output_file
python3 VIFB/a_VIFB_metrics_Qcb.py >> $output_file
python3 VIFB/a_VIFB_metrics_Qcv.py >> $output_file
python3 VIFB/a_VIFB_metrics_Qviff.py >> $output_file
python3 VIFB/a_VIFB_metrics_SD.py >> $output_file
python3 VIFB/a_VIFB_metrics_SF.py >> $output_file
python3 VIFB/a_VIFB_metrics_SSIM.py >> $output_file
python3 VIFB/a_VIFB_metrics_SCD.py >> $output_file
