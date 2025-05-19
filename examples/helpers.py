
def quantize_unsigned_only(tab):
    return tf.abs(tab)
import numpy as np
import tensorflow as tf

def quantize_integers_only(tab):
    return tf.floor(tab + 0.5) # Not using tf.round because "Rounds half to even. Also known as bankers rounding..."

def tab_to_pretty_str(tab, desc=""):
    return f"{desc}: " + " ".join([f"{x:2.10f}".rjust(13) for x in tab])

def print_compared_to_each(a_tab, b_tab, a_tab_desc="", b_tab_desc="", c_tab=None, c_tab_desc=""):
    assert(len(a_tab) == len(b_tab)) # can only compare a with b if the length is equal
    desc_padding = max(len(a_tab_desc), len(b_tab_desc))
    desc_padding = max(desc_padding, len(c_tab_desc))
    a_tab_desc = a_tab_desc.ljust(desc_padding)
    b_tab_desc = b_tab_desc.ljust(desc_padding)
    c_tab_desc = c_tab_desc.ljust(desc_padding)
    print(tab_to_pretty_str(a_tab, a_tab_desc))
    print(tab_to_pretty_str(b_tab, b_tab_desc))
    if c_tab is not None:
        assert (len(b_tab) == len(c_tab))
        print(tab_to_pretty_str(c_tab, c_tab_desc))


# Some numbers in linear space
linear_numbers = tf.constant([-2.3, -1.0, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.5])


