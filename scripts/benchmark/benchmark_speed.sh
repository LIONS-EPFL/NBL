#!/usr/bin/bash

model_path="/path/to/llama_model"
save_file="speedtest/speed_nbl_12.csv"
model_type="normal" # normal or quantized

CUDA_VISIBLE_DEVICES="6" python benchmark_speed3.py \
  --model_path $model_path \
  --model_type ${model_type} \
  --save_file $save_file \