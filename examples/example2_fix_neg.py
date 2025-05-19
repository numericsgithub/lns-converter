import numpy as np
import tensorflow as tf
from helpers import *

print("Lets fix negative numbers!")
# Just make all linear numbers positive using the absolute function tf.abs() #praise_the_absolute
logarithmic_numbers = tf.math.log(tf.abs(linear_numbers)) / tf.math.log(2.0)
# Keep a memory of what was negative and what was positive
linear_numbers_signs = tf.where(linear_numbers < 0.0, -1.0, 1.0)
print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "logarithmic", linear_numbers_signs, "signs")
print()

# Now we just convert it (b) right back to linear space (a) like:   2^(b)=a
# But we also put the signs (s) back into our linear space (a) like: 2^(b)*s=a
print("Convert back to the linear domain (\"recreate\" the linear numbers)")
recreated_linear_numbers = tf.pow(2.0, logarithmic_numbers) * linear_numbers_signs
print_compared_to_each(linear_numbers, recreated_linear_numbers, "linear", "recreated linear")
# Well, now we see the damage we have done! Negative numbers are nan, but tensorflow does handle the -inf case well. Lucky!
# But a closer look shows, we got some errors here and there! Small perturbations! But those do not matter now ...
print()

print("This makes sense! With     2^((+-)b)=a   The result (a) is always positive. A positive (b) represents numbers >1 and a negative (b) represents numbers <1 and (b)==0 representing (a)==1")
print("We extended this now to    (+-)2^((+-)b)=a   Now, the result (a) can also be negative.")

