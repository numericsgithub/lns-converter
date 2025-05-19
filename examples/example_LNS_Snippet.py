import os

from tqdm import tqdm

#export PYTHONPATH=$PYTHONPATH:/home/alex/Dev/LNS_ML_project/lns_converter/examples/lnslinear

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import time
import numpy as np
np.set_printoptions(suppress=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import lnslinear.py.network.net as net
from lnslinear.py.network.LNSLinear import LNSLinear
import tensorflow as tf

input_msb = 2
input_lsb = -1
sum_lsb = -7
use_bias = False

my_lin = LNSLinear(5, 5, input_msb, input_lsb, sum_lsb, bias=use_bias, is_last_layer=False)
my_lin.weight = torch.nn.Parameter(torch.asarray(
    [
        [-0.2, -0.2, -0.2, -0.2, -0.2],
        [-0.1, -0.1, -0.1, -0.1, -0.1],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]))
my_lin.convert()
#print("WEIGHTS ARE", my_lin.weight)

inputs = torch.ones(1, 5, dtype=torch.int)
inputs = torch.asarray([[0.0, 0.1, 0.2, 0.25, 0.3]])
#print("INPUTS ARE", inputs.numpy())
inputs = inputs.clone().apply_(my_lin.xlin2log)
#print("LOG INPUTS ARE", inputs.numpy())
result = my_lin(inputs) #()
#print("LOG OUTPUTS ARE", result)
result = torch.asarray(result.clone(), dtype=torch.float32).apply_(my_lin.xlog2lin)
print("OUTPUTS ARE", result)
exit(0)