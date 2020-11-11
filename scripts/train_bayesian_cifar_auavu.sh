#!/bin/bash
model=resnet20
mode='train'
batch_size=107
lr=0.001189
moped=False

python src/main_bayesian_cifar_auavu.py --lr=$lr --arch=$model --mode=$mode --batch-size=$batch_size --moped=$moped
