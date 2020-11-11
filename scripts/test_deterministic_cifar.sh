#!/bin/bash
model=resnet20
mode='test'
batch_size=10000

python src/main_deterministic_cifar.py --arch=$model --mode=$mode --batch-size=$batch_size
