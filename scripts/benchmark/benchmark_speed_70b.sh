#!/usr/bin/bash


#model_path="mistralai/Mistral-7B-v0.1"
model_path="/path/to/llama-70b_model"
#"mistralai/Mistral-7B-v0.1"
save_file="speedtest/speed_nbl_48.csv"
model_type="quantized" # normal or quantized

CUDA_VISIBLE_DEVICES="1" python benchmark_speed.py \
  --model_path $model_path \
  --model_type ${model_type} \
  --save_file ${save_file} \
  --pretrained

