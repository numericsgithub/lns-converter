import numpy as np
import tensorflow as tf

# Helper function to print original vs reconstructed values side-by-side
def print_compared_to_each(orig, recon, label1, label2, c_tab=None, c_tab_desc=None):
    print(f"\n{label1:15} | {label2:15}", end="")
    if c_tab is not None:
        print(f" | {c_tab_desc}")
    else:
        print()
    print("-" * 60)
    for i in range(orig.shape[0]):
        # Convert tensor to numpy if needed for printing
        o_val = orig[i].numpy() if hasattr(orig[i], 'numpy') else orig[i]
        r_val = recon[i].numpy() if hasattr(recon[i], 'numpy') else recon[i]
        print(f"{o_val:15.10f} | {r_val:15.10f}", end="")
        if c_tab is not None:
            c_val = c_tab[i].numpy() if hasattr(c_tab[i], 'numpy') else c_tab[i]
            print(f" | {c_val:15.10f}")
        else:
            print()

# Helper to format tensor values as a pretty string for printing
def tab_to_pretty_str(tensor, label):
    arr = tensor.numpy() if hasattr(tensor, "numpy") else tensor
    s = f"{label}: "
    s += ", ".join(f"{v:.10f}" for v in arr)
    return s


class Log2QuantizerTF:
    def __init__(self, msb, lsb, fmt):
        self.msb = msb               # Most significant bit position
        self.lsb = lsb               # Least significant bit position (negative for fractional bits)
        self.format = fmt.lower()    # Format string 'sfix' or 'ufix'
        if self.format not in ['sfix', 'ufix']:
            raise ValueError("Format must be 'sfix' or 'ufix'")
        self.frac_bits = abs(lsb)    # Number of fractional bits for quantization
    
    def quantize(self, linear_vals):
        # Convert input list to TensorFlow float32 tensor
        linear_vals = tf.constant(linear_vals, dtype=tf.float32)
        
        if self.format == 'sfix':
            # Signed fixed-point quantization
            return self._quantize_sfix(linear_vals)
        else:
            # Unsigned fixed-point quantization
            return self._quantize_ufix(linear_vals)

    def _quantize_sfix(self, linear_vals):
        # Compute base-2 log of absolute values and extract signs (+/-)
        log_vals = tf.math.log(tf.abs(linear_vals)) / tf.math.log(2.0)
        signs = tf.where(linear_vals < 0, -1.0, 1.0)
        
        #print_compared_to_each(linear_vals, log_vals, "linear", "log2(abs(linear))", signs, "signs")
        
        # Quantize log values by scaling, rounding, and rescaling
        scale_factor = 2 ** self.frac_bits
        q_log = tf.floor(log_vals * scale_factor + 0.5) / scale_factor
        
        # Define clipping range for quantized log values based on fixed-point format
        min_val = -(2 ** (self.msb - 1))
        max_val = 2 ** (self.msb - 1) - 2 ** self.lsb
        
        q_log = tf.clip_by_value(q_log, min_val, max_val)
        print(tab_to_pretty_str(q_log, "Quantized log2 values (sfix format)"))
        
        # Reconstruct linear values from quantized logs and signs
        recreated = tf.pow(2.0, q_log) * signs
        print_compared_to_each(linear_vals, recreated, "original linear", "recreated", q_log, "quantized log")
        return recreated

    def _quantize_ufix(self, linear_vals):
        # Clip input to avoid log(0) and negative values (ufix only supports positives)
        clipped = tf.clip_by_value(linear_vals, 0.001, np.inf)
        print(tab_to_pretty_str(clipped, "ufix clipped linear input"))
    
        # Compute base-2 log of clipped values and take absolute value (ufix non-negative)
        log_vals = tf.math.log(clipped) / tf.math.log(2.0)
        log_vals = tf.abs(log_vals)
    
        # Quantize log2 values using fractional bits with rounding
        scale_factor = 2 ** self.frac_bits
        q_log = tf.floor(log_vals * scale_factor + 0.5) / scale_factor
    
        # Clip quantized values to representable range for ufix format <msb, lsb>
        min_val = 0
        max_val = 2 ** self.msb - 2 ** self.lsb
        q_log = tf.clip_by_value(q_log, min_val, max_val)
    
        print(tab_to_pretty_str(q_log, "Quantized log2 values (ufix format)"))
    
        # Reconstruct linear values from quantized log2 by exponentiating negative q_log
        recreated = tf.pow(2.0, -q_log)
    
        print_compared_to_each(clipped, recreated, "ufix linear", "recreated ufix", q_log, "quantized log (ufix)")
        return recreated


def main():
    # User input: linear values as comma separated floats
    raw_vals = input("Enter comma-separated linear values (e.g. -2.3, -1.0, 0.0, 0.1, 0.2): ")
    linear_vals = [float(v.strip()) for v in raw_vals.split(",") if v.strip()]
    
    # User input: quantization format (sfix or ufix)
    fmt = input("Choose format (sfix or ufix): ").strip().lower()
    # User input: MSB and LSB positions for fixed-point format
    msb = int(input("Enter MSB position (e.g. 3): ").strip())
    lsb = int(input("Enter LSB position (e.g. -1): ").strip())
    
    # Create quantizer instance and run quantization
    quantizer = Log2QuantizerTF(msb, lsb, fmt)
    quantizer.quantize(linear_vals)

if __name__ == "__main__":
    main()
