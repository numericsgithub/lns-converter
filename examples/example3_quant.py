import numpy as np
import tensorflow as tf
from helpers import *

print("So, we fixed negative numbers! (This is based on example2)")
print("The goal is to save some bits and bytes to store the logarithmic representation.")
print("All weights are <=1, so the sign bit is not needed! Lets kick it out and look at the damage!")
# Same procedure as before. No change here...
logarithmic_numbers = tf.math.log(tf.abs(linear_numbers)) / tf.math.log(2.0)
linear_numbers_signs = tf.where(linear_numbers < 0.0, -1.0, 1.0)
print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "logarithmic", linear_numbers_signs, "signs")
print()


# Ok, now we destroy stuff! ... I mean quantize!
# We take the absolute value and smack a negative sign in front of everything.
logarithmic_numbers = -tf.abs(logarithmic_numbers) # kick out the sign bit here!
print(tab_to_pretty_str(logarithmic_numbers, "Quantized logarithmic values"))
print()
# Oh... this will be nasty!

print("Convert back to the linear domain (\"recreate\" the linear numbers)")
recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs
print_compared_to_each(linear_numbers, recreated_linear_numbers, "linear", "recreated linear")
print()

print("Well it works fine! Just not so fine for those numbers >1. But who cares?")


