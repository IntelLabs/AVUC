#!/bin/bash
model=resnet50
mode='test'
val_batch_size=2500
num_monte_carlo=128

python src/main_bayesian_imagenet_avu.py data/imagenet --arch=$model --mode=$mode --val_batch_size=$val_batch_size --num_monte_carlo=$num_monte_carlo
