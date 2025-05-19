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


# todo Quantisiere inputs (ohne sign bit, Also nur [0; 1]) 
# warum Inputs ohne Sign Bit? 
def quantize_inputs(lineare_input, msb_pos, lsb_pos):
    # kicks out numbers that are too big or too small (< 0 or > 1)
    lineare_input = tf.where(lineare_input < 0, 0, tf.where(lineare_input > 1, 1.0, lineare_input))
    # brings lineare numbers to logarithmic space
    logarithmic_input = tf.math.log(tf.abs(lineare_input)) / tf.math.log(2.0)
    # 
    logarithmic_input = -tf.abs(logarithmic_input)

    # scales the numbers in the given format using scale factor and clipping
    frac_bits = abs(lsb_pos)
    scale_factor = 2 ** (frac_bits)
    logarithmic_input = tf.floor((logarithmic_input) * scale_factor + 0.5) / scale_factor
    min_lns_value = -2 ** (msb_pos + 1) + 2 ** lsb_pos
    max_lns_value = 2 ** (msb_pos + 1) - 2 ** lsb_pos 
    logarithmic_input = tf.clip_by_value(logarithmic_input, min_lns_value, max_lns_value) 

    return logarithmic_input

# todo Quantisiere gewichte (mit sign bit, Also nur [-1;1] also genauso wie zuvor)
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

def reconstruct_weights(logarithmic_numbers, linear_numbers_signs):
    return tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs


# todo Implementiere eine funktion die gewichte und input verrechnet wie in einem Dense layer.
# todo Dafür kann tensorflow (tf.keras.layers.Dense) benutzen
# todo einfach in chatgpt rein

# compute output like a dense layer with given inputs and weights (using keras)
def compute_dense(inputs, weights):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(weights.shape[1], input_shape=(inputs.shape[1],)))
    bias = np.zeros((weights.shape[1],))
    model.layers[0].set_weights([weights.numpy(), bias])
    output = model.predict(inputs)
    return output

def main():
    input_msb = 2
    input_lsb = -1
    sum_lsb = -7
    use_bias = False

    my_lin = LNSLinear(5, 5, input_msb, input_lsb, sum_lsb, bias=use_bias)
    my_lin.weight = torch.nn.Parameter(torch.asarray(
        [
            [-0.2, -0.2, -0.2, -0.2, -0.2],
            [-0.1, -0.1, -0.1, -0.1, -0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.4, 0.4, 0.4, 0.4, 0.4],
        ]))
    print("WEIGHTS ARE \n", my_lin.weight)

    inputs = torch.ones(1, 5, dtype=torch.int)
    inputs = torch.asarray([[0.0, 0.1, 0.2, 0.25, 0.3]])
    print("INPUTS ARE \n", inputs.numpy())

    # original method 
    my_lin_copy = my_lin.weight.clone()
    my_lin.convert()
    print("\n CONVERTED WEIGHTS ARE \n", my_lin.weight)

    # new method
    weights_quantized, linear_numbers_signs = quantize_weights(my_lin_copy.detach().numpy(), input_msb, input_lsb)
    print("\n QUANTIZED LNS WEIGHTS ARE \n", weights_quantized.numpy())

    inputs_quantized = quantize_inputs(inputs, input_msb, input_lsb)
    print("\n QUANTIZED LNS INPUTS ARE \n", inputs_quantized.numpy())

    weights_reconstructed = reconstruct_weights(inputs_quantized, linear_numbers_signs)

    input_reconstructed = reconstruct_weights(inputs_quantized, np.ones_like(inputs_quantized))
    
    print("\n RECONSTRUCTED WEIGHTS ARE \n", weights_reconstructed.numpy())

    # compute the output of the dense layer
    output = compute_dense(input_reconstructed, weights_reconstructed)
    print("\n OUTPUT IS \n", output)




if __name__ == "__main__":
    main()
