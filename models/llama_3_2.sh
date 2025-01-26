#!/bin/zsh

# variables
model_name="llama3.2"
custom_model_name="llama3.2"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./model_files/llama3.2_ModelFile
