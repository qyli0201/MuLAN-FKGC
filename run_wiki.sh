#!/usr/bin/env bash
set -ex

python main.py  --few 3 --prefix wikilr2e-4noise0_"3shot" --device 'cuda:1' --max_neighbor 30  --lr 2e-4 --batch_size 128  --datapath "data/Wiki/" --hidden_size 50 --fine_tune --max_batches 300000 --noise_rate 0.0


