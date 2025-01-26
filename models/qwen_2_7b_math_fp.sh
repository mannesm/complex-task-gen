#!/bin/zsh

# variables
model_name="qwen2-math:7b-instruct-fp16"
custom_model_name="qwen2_7b_math_fp"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./model_files/qwen2_7b_math_fp_ModelFile
