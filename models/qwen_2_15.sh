#!/bin/zsh

# variables
model_name="qwen2.5:1.5b"
custom_model_name="qwen2.5:1.5b"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./model_files/qwen2_7b_math_fp.ModelFile
