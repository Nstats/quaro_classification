#!/usr/bin/env bash
python train.py --q_max_len=25 --dp_keep_prob=0.8
python train.py --q_max_len=50 --dp_keep_prob=0.8
python train.py --q_max_len=75 --dp_keep_prob=0.8
python train.py --q_max_len=100 --dp_keep_prob=0.8

python train.py --q_max_len=25 --dp_keep_prob=0.5
python train.py --q_max_len=50 --dp_keep_prob=0.5
python train.py --q_max_len=75 --dp_keep_prob=0.5
python train.py --q_max_len=100 --dp_keep_prob=0.5

python train.py --q_max_len=25 --dp_keep_prob=0.1
python train.py --q_max_len=50 --dp_keep_prob=0.1
python train.py --q_max_len=75 --dp_keep_prob=0.1
python train.py --q_max_len=100 --dp_keep_prob=0.1
