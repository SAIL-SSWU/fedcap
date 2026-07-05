#!/bin/bash

MODEL_PATH="/home/dlgkrud/fedcap/fed_runs/fedavg_cifar100_P100_R250_20260324-033438/global_model_final.pth"
LOGDIR="/home/dlgkrud/fedcap/fedcap/logs"

for KP in 1 3 5 10 
do
  echo "Running Kp=${KP}"

  python main.py \
    --alg fedcap \
    --dataset cifar100 \
    --model resnet18-cifar100 \
    --partition noniid \
    --beta 0.5 \
    --logdir "${LOGDIR}" \
    --out_dim 256 \
    --n_parties 100 \
    --sample_fraction 0.2 \
    --comm_round 0 \
    --epochs 10 \
    --batch-size 64 \
    --optimizer sgd \
    --lr 0.01 \
    --device cuda:0 \
    --init_seed 0 \
    --mu 0 \
    --temperature 0 \
    --use_project_head 0 \
    --Kg 0 \
    --head_lr 0.001 \
    --Kp ${KP} \
    --load_model_file "${MODEL_PATH}" \
    > "${LOGDIR}/kp_${KP}.log" 2>&1
done