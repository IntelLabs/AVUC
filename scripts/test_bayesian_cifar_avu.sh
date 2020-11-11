#!/bin/bash
model=resnet20
mode='test'
batch_size=10000
num_monte_carlo=128

python src/main_bayesian_cifar_avu.py --arch=$model --mode=$mode --batch-size=$batch_size --num_monte_carlo=$num_monte_carlo
