import numpy as np
import tensorflow as tf
from helpers import *  # Assumes print_compared_to_each and tab_to_pretty_str are defined

# === PARAMETERS FOR FIXED-POINT FORMAT ===
msb_pos = 3   # Most significant bit position (integer part of fixed-point format)
lsb_pos = -1  # Least significant bit position (fractional part of fixed-point format)

# === STEP 1: USE PRECISE INPUT VALUES ===
# Define a list of signed input values, including negative, zero, and positive floats
linear_numbers = tf.constant([
    -2.2999999523, -1.0000000000, -0.1000000015, 0.0000000000, 0.1000000015,
    0.2000000030, 0.3000000119, 0.5000000000, 0.6999999881, 0.8000000119,
    0.8999999762, 1.0000000000, 1.1000000238, 1.5000000000
], dtype=tf.float32)

# === STEP 2: COMPUTE LOG2(ABS(x)) AND TRACK SIGNS ===
# Calculate the base-2 logarithm of absolute input values
# Also track the sign of each original input value for later reconstruction
logarithmic_numbers = tf.math.log(tf.abs(linear_numbers)) / tf.math.log(2.0)
linear_signs = tf.where(linear_numbers < 0, -1.0, 1.0)

# Print original values, log2 values, and signs
print_compared_to_each(linear_numbers, logarithmic_numbers, "linear", "log2(abs(linear))", linear_signs, "signs")

# === STEP 3: FIXED-POINT QUANTIZATION (SFIX FORMAT) ===
# Use LNS-style: negate the magnitude of the log2 values
# logarithmic_numbers = -tf.abs(logarithmic_numbers)

# Quantize the values using the fixed-point format
frac_bits = abs(lsb_pos)
scale_factor = 2 ** frac_bits
quantized_log = tf.floor(logarithmic_numbers * scale_factor + 0.5) / scale_factor  # Round and rescale

# Clip to fixed-point representable range (based on sfix<msb_pos, frac_bits>)
min_val = -(2 ** (msb_pos - 1))
max_val = 2 ** (msb_pos - 1) - 2 ** lsb_pos
quantized_log = tf.clip_by_value(quantized_log, min_val, max_val)
print("min_val",min_val, msb_pos, lsb_pos)
print("max_val",max_val)
# Print quantized log2 values in sfix format
print(tab_to_pretty_str(quantized_log, "Quantized log2 values (sfix format)"))

# === STEP 4: RECONSTRUCT LINEAR VALUES FROM QUANTIZED LOG2 ===
# Recreate the linear approximation using 2^(quantized_log) * sign
recreated_linear = tf.pow(2.0, quantized_log) * linear_signs

# Print original vs reconstructed values
print_compared_to_each(linear_numbers, recreated_linear, "original linear", "recreated", c_tab=quantized_log, c_tab_desc="quantized log")

# === STEP 5: UFIX FORMAT (Unsigned, only positive values) ===
print("\nNow repeat with ufix<4,2> (unsigned, no negatives allowed)")

ufix_msb = 3
ufix_lsb = -2

# Clip input to [0.001, ∞) to avoid log(0) and disallow negative inputs for ufix
linear_numbers_ufix = tf.clip_by_value(linear_numbers, 0.001, np.inf)

# Compute base-2 logarithm of clipped values
log2_ufix = tf.math.log(linear_numbers_ufix) / tf.math.log(2.0)
logarithmic_numbers_ufix = tf.abs(log2_ufix)

# Quantize the log2 values using ufix format
frac_bits = abs(ufix_lsb)
scale_factor = 2 ** frac_bits
quantized_ufix = tf.floor(logarithmic_numbers_ufix * scale_factor + 0.5) / scale_factor

# Clip to representable range for ufix<ufix_msb, frac_bits>
min_val_ufix = 0
max_val_ufix = 2 ** (ufix_msb) - 2 ** ufix_lsb
quantized_ufix = tf.clip_by_value(quantized_ufix, min_val_ufix, max_val_ufix)
print("min_val",min_val_ufix, ufix_msb, ufix_lsb)
print("max_val",max_val_ufix)

# Reconstruct the linear values from quantized log2 values
recreated_linear_ufix = tf.pow(2.0, -quantized_ufix)

# Print original vs reconstructed ufix results
print_compared_to_each(linear_numbers_ufix, recreated_linear_ufix, "ufix linear", "recreated ufix", c_tab=quantized_ufix, c_tab_desc="quantized log (ufix)")


