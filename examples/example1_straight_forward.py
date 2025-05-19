import numpy as np
import tensorflow as tf
from helpers import *

# Let us convert those linear numbers (a) to logarithmic (b) like:   2^(b)=a
# So: b = log(a) / log(2)
print("Convert to the logarithmic domain")
logarithmic_numbers = tf.math.log(linear_numbers) / tf.math.log(2.0)
print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "logarithmic")
# It works! Kind of ... it does not work with 0 and with negative numbers
# So only numbers >0 can be converted this way.
# But even worse! numbers >1 are positive numbers and numbers <1 are negative numbers in the log domain!!!
# So we would need to save the sign bit too there!
print()

# Now we just convert it (b) right back to linear space (a) like:   2^(b)=a
print("Convert back to the linear domain (\"recreate\" the linear numbers)")
recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers)
print_compared_to_each(linear_numbers, recreated_linear_numbers, "linear", "recreated linear")
# Well, now we see the damage we have done! Negative numbers are nan, but tensorflow does handle the -inf case well. Lucky!
# But a closer look shows, we got some errors here and there! Small perturbations! But those do not matter now ...
print()