import numpy as np
import tensorflow as tf
from helpers import *

print("So, lets go even further! (based on the example before)")
print("Lets round every logarithmic number to the next integer! What is the worst that could happen?")
# Same procedure as before. No change here...
logarithmic_numbers = tf.math.log(tf.abs(linear_numbers)) / tf.math.log(2.0)
linear_numbers_signs = tf.where(linear_numbers < 0.0, -1.0, 1.0)
print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "logarithmic", linear_numbers_signs, "signs")
print()


# Ok, now we destroy stuff! ... I mean quantize! EVEN MORE!
logarithmic_numbers = -tf.abs(logarithmic_numbers) # kick out the sign bit here! Just like before ...
logarithmic_numbers = tf.floor(logarithmic_numbers + 0.5) # not using tf.round because "Rounds half to even. Also known as bankers rounding..."
print(tab_to_pretty_str(logarithmic_numbers, "Quantized logarithmic values"))
print()
# Oh... this will be nasty!

print("Convert back to the linear domain (\"recreate\" the linear numbers)")
# Same procedure as before. No change here...
recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs
print_compared_to_each(linear_numbers, recreated_linear_numbers, "linear", "recreated linear")
print()

print("It does work! There are some losses here and there...")


