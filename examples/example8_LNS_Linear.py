import os
from tqdm import tqdm
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


# quantize inputs (only [0; 1])
def quantize_inputs(lineare_input, msb_pos, lsb_pos):
    # kicks out numbers that are too big or too small (< 0 or > 1)
    lineare_input = tf.where(lineare_input < 0, 0, tf.where(lineare_input > 1, 1.0, lineare_input))
    # brings lineare numbers to logarithmic space
    logarithmic_input = tf.math.log(tf.abs(lineare_input)) / tf.math.log(2.0)
    logarithmic_input = -tf.abs(logarithmic_input)

    # scales the numbers in the given format using scale factor and clipping
    frac_bits = abs(lsb_pos)
    scale_factor = 2 ** (frac_bits)
    logarithmic_input = tf.floor((logarithmic_input) * scale_factor + 0.5) / scale_factor
    min_lns_value = -2 ** (msb_pos + 1) + 2 ** lsb_pos
    max_lns_value = 2 ** (msb_pos + 1) - 2 ** lsb_pos 
    logarithmic_input = tf.clip_by_value(logarithmic_input, min_lns_value, max_lns_value) 

    return logarithmic_input

# quantize weights ([-1;1])
def quantize_weights(lineare_weights, msb_pos, lsb_pos):
    # kicks out numbers that are too big or too small (< -1 or > 1)
    lineare_weights = tf.where(lineare_weights < -1, -1.0, tf.where(lineare_weights > 1, 1.0, lineare_weights))
    # brings lineare numbers to logarithmic space
    logarithmic_weights = tf.math.log(tf.abs(lineare_weights)) / tf.math.log(2.0)
    # saves the signs for reconstruction
    logarithmic_numbers_signs = tf.where(lineare_weights < 0.0, -1.0, 1.0)
    # all negative 
    logarithmic_weights = -tf.abs(logarithmic_weights)
    # scales the numbers in the given format using scale factor and clipping
    frac_bits = abs(lsb_pos)
    scale_factor = 2 ** (frac_bits)
    logarithmic_weights = tf.floor((logarithmic_weights) * scale_factor + 0.5) / scale_factor
    min_lns_value = -2 ** (msb_pos + 1) + 2 ** lsb_pos
    max_lns_value = 2 ** (msb_pos + 1) - 2 ** lsb_pos 
    logarithmic_weights = tf.clip_by_value(logarithmic_weights, min_lns_value, max_lns_value) 

    return logarithmic_weights, logarithmic_numbers_signs

# reconstruct weights using the logarithmic numbers and the saved signs
def reconstruct_weights(logarithmic_numbers, linear_numbers_signs):
    return tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs


# compute output like a dense layer with given inputs and weights (using keras)
def compute_dense(inputs, weights):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(weights.shape[0], input_shape=(weights.shape[1],)))
    # print("\n weights.shape[0]  is \n", weights.shape[0])
    # print(weights.numpy().transpose().shape[0])
    bias = np.zeros((weights.shape[0],))
    model.layers[0].set_weights([weights.numpy().transpose(), bias])
    output = model.predict(inputs)
    return output

# uses cpp method to calculate the output of the LNSLinear layer
def calculate_output_cpp(inputs, my_lin):
    my_lin.convert()  # uses wlin2log 
    inputs = inputs.clone().apply_(my_lin.xlin2log)   # uses xlin2log
    result = my_lin(inputs) 
    return torch.asarray(result.clone(), dtype=torch.float32).apply_(my_lin.xlog2lin) # reconstruct using xlog2lin

# uses python method to calculate the output of the LNSLinear layer     
def calculate_output_py(inputs, weights, input_msb, input_lsb):
    weights_quantized, linear_numbers_signs = quantize_weights(weights.detach().numpy(), input_msb, input_lsb)
    #print("\n QUANTIZED LNS WEIGHTS ARE \n", weights_quantized.numpy())
    inputs_quantized = quantize_inputs(inputs, input_msb, input_lsb)
    #print("\n QUANTIZED LNS INPUTS ARE \n", inputs_quantized.numpy())
    weights_reconstructed = reconstruct_weights(weights_quantized, linear_numbers_signs)
    input_reconstructed = reconstruct_weights(inputs_quantized, np.ones_like(inputs_quantized))
    
    # print("\n RECONSTRUCTED WEIGHTS ARE \n", weights_reconstructed.numpy())
    # print("\n RECONSTRUCTED INPUTS ARE \n", input_reconstructed.numpy())

    # compute the output of the dense layer
    return compute_dense(input_reconstructed, weights_reconstructed)

def main():
    input_msb = 1
    input_lsb = -2
    sum_lsb = -12
    use_bias = False

    weights = torch.asarray(
            [
                [-0.2, 0.8, 0.8, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0, 1.0],
            ])

    my_lin = LNSLinear(weights.shape[1], weights.shape[0], input_msb, input_lsb, sum_lsb, bias=use_bias)
    my_lin.weight = torch.nn.Parameter(weights)
    #print("WEIGHTS ARE \n", my_lin.weight)

    inputs = torch.ones(1, 5, dtype=torch.int)
    inputs = torch.asarray([[0.0, 0.1, 0.2, 0.25]])
    #print("INPUTS ARE \n", inputs.numpy())

    # copy of the weights and inputs for the python conversion
    my_lin_weights_copy = my_lin.weight.clone()
    my_input_copy = inputs.clone()


    # old method using cpp 
    result_cpp = calculate_output_cpp(inputs, my_lin)
    print("\n OUTPUT CPP IS: \n", result_cpp)

    # new method using only python
    result_py = calculate_output_py(my_input_copy, my_lin_weights_copy, input_msb, input_lsb)
    print("\n OUTPUT PY IS: \n", result_py)



if __name__ == "__main__":
    main()
