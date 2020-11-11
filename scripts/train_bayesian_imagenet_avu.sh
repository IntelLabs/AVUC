#!/bin/bash
model=resnet50
mode='train'
batch_size=384
lr=0.001
moped=True

python -u src/main_bayesian_imagenet_avu.py data/imagenet --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --moped=$moped
