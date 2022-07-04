#!/bin/bash

project="quantized_cnn_v2_try_7"
model="pruned_cnn_model_v4"

faketime -f '-1y' python3 build_model.py $model $project
