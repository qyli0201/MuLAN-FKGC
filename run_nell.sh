#!/usr/bin/env bash
set -ex

python main.py  --few 3 --prefix nelllr8e-5noise0_"3shot" --device 'cuda:0' --max_neighbor 30  --lr 8e-5 --batch_size 128 --fine_tune --noise_rate 0.0


