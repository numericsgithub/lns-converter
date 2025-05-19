import tensorflow as tf 

"""
Converts a lin float to an lns representation as float
Args:
    lin_value: 
    msb_bit_pos: 
    lsb_bit_pos: 

Returns: LNS representation of the lin value

"""
def lin_to_log(lin_value:float, msb_bit_pos:int, lsb_bit_pos:int) -> float: # -> log_value_result

    # calculate the maximum and minimum representable values for the given number of bits
    min_lns_value = 2**lsb_bit_pos
    max_lns_value = (2**msb_bit_pos - 1 + 2**lsb_bit_pos)
    print("max_value", 2**max_lns_value)
    print("min_value", min_lns_value)

    # Check if the integer is within the representable range
    if lin_value < min_lns_value: 
        return 0.0
    elif lin_value > 2**max_lns_value:
        return max_lns_value

    # input to tf.constant for tf functions
    lin_value = tf.constant(lin_value, dtype=tf.float32)
    
    # Calculate the logarithm value (no log(2) so --> log10/log2 )
    log_value = tf.math.log(tf.abs(lin_value)) / tf.math.log(2.0)

    # Calculate the number of fractional bits  
    frac_bits = abs(lsb_bit_pos)
    
    # Scale the logarithm value
    scale_factor = 2 ** frac_bits
    log_value_scaled = tf.round(log_value * scale_factor)
    
    # Scale back to the fixed-point representation
    log_value_result = log_value_scaled / scale_factor

    return log_value_result.numpy()


"""
Converts an lns float back to lin space
Args:
    log_value_result: 
    msb_bit_pos: 
    lsb_bit_pos: 

Returns:

"""
def log_to_lin(log_value_result:float, msb_bit_pos:int, lsb_bit_pos:int) -> float: # -> lin_value_new

    pass


#lin_value = tf.constant([1.0, 0.5, ...])
# lin_value = lin_value + 0.1
# lin_value = lin_value + tf.random.uniform(-0.1, 0.1, tf.shape(lin_value))

#lin_value = 0.5
#log_value_result = lin_to_log(lin_value) # 1.0 (2^-1)
#lin_value_new = log_to_lin(log_value_result) # 0.5

lin_value = 8
msb_bit_pos = 3
lsb_bit_pos = -1
log_value_result = lin_to_log(lin_value, msb_bit_pos, lsb_bit_pos)

print("Linear Value:", lin_value)
print("Logarithmic Value:", log_value_result)