#!/bin/bash
set -euo pipefail

echo ">>> Running TrainLeNet.py with LNS settings..."

python3 ./TrainLeNet.py \
  -tt lns \
  -qd layer-wise \
  -b-bits 10 \
  -w-bits 3 \
  -a-bits 3 \
  --lns-format sfix \
  -lr 0.0001 \
  -no-skip \
  -desc test5 \
  --checkpoint "data/training/LeNetLike/float/test5/model_best_q.npz"
