import numpy as np
import tensorflow as tf
from helpers import *

msb_pos = 3
lsb_pos = -1

# todo Alle Zahlen die kleiner -1 sind müssen garantiert auf -1 "gerundet" werden. Genauso mit Zahlen größer 1
linear_numbers = tf.where(linear_numbers < -1, -1.0, tf.where(linear_numbers > 1, 1.0, linear_numbers))
print("Next step!")
print("The logarithmic number has to be a fixed point number!")
# Same procedure as before. No change here...
logarithmic_numbers = tf.math.log(tf.abs(linear_numbers)) / tf.math.log(2.0)
linear_numbers_signs = tf.where(linear_numbers < 0.0, -1.0, 1.0)
# need to keep track of signs of the logarithmic numbers as well

print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "logarithmic", linear_numbers_signs, "signs")

print()


# Ok, now we destroy stuff! ... I mean quantize! EVEN MORE!
logarithmic_numbers = -tf.abs(logarithmic_numbers) # kick out the sign bit here! Just like before ...

# Calculate the number of fractional bits
frac_bits = abs(lsb_pos)
scale_factor = 2 ** (frac_bits)
# quantize the logarithmic numbers using the scale factor
logarithmic_numbers = tf.floor((logarithmic_numbers) * scale_factor + 0.5) / scale_factor

# calculate the maximum and minimum representable values for the given number of bits
min_lns_value = -2 ** (msb_pos + 1) + 2 ** lsb_pos
max_lns_value = 2 ** (msb_pos + 1) - 2 ** lsb_pos
print("min_lns_value", min_lns_value)
print("max_lns_value", max_lns_value)
# clips the values to the representable range
logarithmic_numbers = tf.clip_by_value(logarithmic_numbers, min_lns_value, max_lns_value)

# print like before
print(tab_to_pretty_str(logarithmic_numbers, "Quantized logarithmic values"))
print()

print("Convert back to the linear domain (\"recreate\" the linear numbers)")
# convert back to linear (added the signs in the exponent)
#recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers * logarithmic_numbers_signs) * linear_numbers_signs
recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs
print_compared_to_each(linear_numbers, recreated_linear_numbers, "linear", "recreated linear", c_tab=logarithmic_numbers, c_tab_desc="logarithmic_numbers")
print()

print("It does work! There are some losses here and there...")

# TODO Check if the results match with the results from the original code!
