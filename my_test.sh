#!/usr/bin/env bash
python train.py --q_max_len=25 --dp_keep_prob=0.2 --ratio_1_0=2
python train.py --q_max_len=25 --dp_keep_prob=0.1 --ratio_1_0=2
python train.py --q_max_len=100 --dp_keep_prob=0.2 --ratio_1_0=2
python train.py --q_max_len=100 --dp_keep_prob=0.1 --ratio_1_0=2
