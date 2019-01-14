#!/usr/bin/env bash
python train.py --q_max_len=25 --dp_keep_prob=0.1 --lr=0.001
python train.py --q_max_len=25 --dp_keep_prob=0.1 --lr=0.01
python train.py --q_max_len=25 --dp_keep_prob=0.8 --lr=0.001
python train.py --q_max_len=25 --dp_keep_prob=0.8 --lr=0.01

python train.py --q_max_len=50 --dp_keep_prob=0.1 --lr=0.001
python train.py --q_max_len=50 --dp_keep_prob=0.1 --lr=0.01
python train.py --q_max_len=50 --dp_keep_prob=0.8 --lr=0.001
python train.py --q_max_len=50 --dp_keep_prob=0.8 --lr=0.01
