#import tensorflow as tf
import numpy as np

def convert_to_lns(integer_value, msb_position, lsb_position):

    # calculate the maximum and minimum representable values for the given number of bits
    max_lns_value = 2**msb_position - 1 + 2**lsb_position  
    min_lns_value = 2**lsb_position  

    # Check if the integer is within the representable range
    if integer_value < 2**min_lns_value or integer_value > 2**max_lns_value:
        raise ValueError("Integer is out of representable bit range")
    
    # integer to logarithmic value 
    log_value = np.log2(integer_value)

    # Represent the logarithmic value in binary format with given bits
    int_part = int(np.floor(log_value))
    frac_part = log_value - int_part
    
    # Initialize bit representation
    bits = np.zeros(msb_position - lsb_position + 1, dtype=int)
    
    # Convert the integer part
    for i in range(msb_position, lsb_position - 1, -1):
        if int_part >= 2**i:
            bits[msb_position-i] = 1
            int_part -= 2**i
    
    # Convert the fractional part
    if frac_part >= 2**lsb_position:
        bits[-1] = 1
    
    return bits

integer = 12
msb = 3
lsb = -1 
bits = convert_to_lns(integer, msb, lsb)

print("Integer:", integer)
print("LNS Bits:", bits)



def lin_to_log(lin_value:float, msb_bit_pos:int, lsb_bit_pos:int) -> float: # -> log_value_result
    """
    Converts a lin float to an lns representation as float
    Args:
        lin_value: 
        msb_bit_pos: 
        lsb_bit_pos: 

    Returns:

    """
    pass

def log_to_lin(log_value_result:float, msb_bit_pos:int, lsb_bit_pos:int) -> float: # -> lin_value_new
    """
    Converts an lns float back to lin space
    Args:
        log_value_result: 
        msb_bit_pos: 
        lsb_bit_pos: 

    Returns:

    """
    pass


# lin_value = tf.constant([1.0, 0.5, ...])
# lin_value = lin_value + 0.1
# lin_value = lin_value + tf.random.uniform(-0.1, 0.1, tf.shape(lin_value))

lin_value = 0.5
log_value_result = lin_to_log(lin_value) # 1.0 (2^-1)
lin_value_new = log_to_lin(log_value_result) # 0.5
