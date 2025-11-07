import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# from codegen.RPAG_graph_parser import generate_rpag_mcm, parse_graph, run_vivado_synthesis
# from codegen.MUXman import quant_weights
from mquat.QuantizerBase import DEFAULT_DATATYPE
from training.TrainingTools import QTraining

import mquat as mq  # noqa: E402
import models.LeNet5_Like as ln5  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
import tensorflow as tf  # noqa: E402

# First train the model without any quantization. Just a normal floating-point 32bit training!
# -tt float -desc test5

# Then, train the model with some quantization (QAT - quantization aware training)
# Replace this fixed-point training with your LNS training.
# Example:
# -tt lns -qd layer-wise -b-bits 5 -w-bits 5 -a-bits 5 --lns-format sfix -lr 0.0001 -no-skip -desc test5


def strict_top1_with_tolerance(tol=1e-4, name="fixed_(strict)_top_1"):
    """
    Tolerance-based strict Top-1 accuracy.
    Returns 0.0 instead of NaN if no values match.
    """
    import tensorflow as tf

    def _metric(y_true, y_pred):
        # tf.print("top debug", y_pred)
        y_true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)  # force int32
        max_val = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        close_mask = tf.less_equal(tf.abs(y_pred - max_val), tol)

        batch_indices = tf.range(tf.shape(y_pred)[0], dtype=tf.int32)
        true_is_close = tf.gather_nd(
            close_mask,
            tf.stack([batch_indices, y_true_idx], axis=1)
        )

        val = tf.reduce_mean(tf.cast(true_is_close, tf.float32))
        # Replace NaN with 0
        return tf.where(tf.math.is_finite(val), val, tf.constant(0.0, dtype=tf.float32))

    _metric.__name__ = name
    return _metric

class TrainLeNet(QTraining):
    def __init__(self, INPUT_SHAPE=[28, 28, 1], NUM_CLASSES=10, batch_size=256):
        super().__init__("LeNetLike", INPUT_SHAPE, NUM_CLASSES, batch_size)

    def get_top1_float(self) -> float:
        return 0.9920

    def load_inital_model(self, model):
        pass

    def apply_ptq(self, model):
        return  # No post training quantization here

    def quantize_model(self, model, selected_train_type, weights_bits_total, bias_bits_total,
                       activation_bits_total, product_bits, adder, std_keep_factor, quantize_actication) -> None:
        args = self.args
        print("SELECTED selected_train_type", selected_train_type, f"and adder is \"{adder}\"")

        def get_quantizer(name, i="all", j="all", no_chan=False):
            suffix = "" if no_chan else f"per_chan_{i}_{j}_f"
            if selected_train_type == "fixed":
                return mq.FlexPointQuantizer(name + suffix, weights_bits_total)
            if selected_train_type == "lns":
                # LNSQuantizer expects an internal quantizer (use FlexPointQuantizer)
                internal_q = mq.FlexPointQuantizer(name + suffix + "_internal", weights_bits_total)
                return mq.LNSQuantizer(name + suffix, internal_q, exponent_dtype=tf.dtypes.float32)
            return mq.AddQuantizer(name + suffix, str(adder) + "Add" + str(weights_bits_total))

        quant_depth = args["quant-depth"]

        def quant_Conv(conv, conv_name, filters, input_channels, use_adder):
            if args["no-weight"] != True:
                if quant_depth == "kernel-wise":
                    conv.f.quant_out = mq.PerKernelQuantizer(f"{conv_name}_kw_f", get_quantizer, 3, filters, 2, input_channels)
                elif quant_depth == "channel-wise":
                    conv.f.quant_out = mq.PerChannelQuantizer(f"{conv_name}_cw_f", get_quantizer, 3, filters)
                elif quant_depth == "layer-wise":
                    conv.f.quant_out = get_quantizer(f"{conv_name}_f", no_chan=True)
                else:
                    raise Exception(f"Unkwon quant_depth \"{quant_depth}\"")
                conv.conv2d.mat_mul.quant_mul = mq.FlexPointQuantizer(f"{conv_name}_products", product_bits)
            if args["no-bias"] != True:
                conv.b.quant_out = mq.FlexPointQuantizer(f"{conv_name}_b", bias_bits_total)
            if args["no-activation"] != True:
                conv.quant_in = get_quantizer(f"{conv_name}_out", no_chan=True)
                conv.quant_out = mq.FlexPointQuantizer(f"output_{conv_name}_out", activation_bits_total)

        def quant_Dense(dense, dense_name, neurons, use_adder):
            if args["no-weight"] != True:
                if quant_depth == "channel-wise" or quant_depth == "kernel-wise":
                    dense.w.quant_out = mq.PerChannelQuantizer(f"{dense_name}_cw_w", get_quantizer, 1, neurons)
                elif quant_depth == "layer-wise":
                    dense.w.quant_out = get_quantizer(f"{dense_name}_w", no_chan=True)
                else:
                    raise Exception(f"Unkwon quant_depth {quant_depth}")
            if args["no-bias"] != True:
                dense.b.quant_out = mq.FlexPointQuantizer(f"{dense_name}_b", bias_bits_total)
            # if args["no-activation"] != True:
            #     dense.quant_in = mq.FlexPointQuantizer(f"{dense_name}_out", activation_bits_total)

        if selected_train_type in ("fixed", "adder", "lns"):
            use_adder = selected_train_type == "adder"
            quant_Conv(model.conv1, "QuantConv1", 8, 1, use_adder, is_first=True)
            quant_Conv(model.conv2, "QuantConv2", 16, 8, use_adder)
            quant_Dense(model.dense1, "QuantDense1", 10, use_adder)
        elif selected_train_type == "float":
            pass
        else:
            raise Exception("UNKNOWN train type!")

    # ---------- Strict metric NaN-safe wrapper ----------
    class SafeStrictMetric(tf.keras.metrics.Metric):
        """
        Wraps either:
          - a Keras Metric (has update_state/result), or
          - a metric function f(y_true, y_pred) -> tensor,
        and guarantees a finite result (NaN/Inf -> 0.0).
        """

        def __init__(self, base, name="fixed (strict) top 1"):
            super().__init__(name=name)
            self.base = base
            # For function-style metrics, aggregate with a Mean
            self._mean = tf.keras.metrics.Mean(name=name + "_mean")

        def update_state(self, y_true, y_pred, sample_weight=None):
            if hasattr(self.base, "update_state"):
                # Metric-style
                return self.base.update_state(y_true, y_pred, sample_weight=sample_weight)
            else:
                # Function-style
                val = self.base(y_true, y_pred)
                val = tf.where(tf.math.is_finite(val), val, tf.zeros_like(val))
                return self._mean.update_state(val, sample_weight=sample_weight)

        def result(self):
            if hasattr(self.base, "result"):
                r = self.base.result()
            else:
                r = self._mean.result()
            return tf.where(tf.math.is_finite(r), r, tf.constant(0.0, dtype=r.dtype))

        def reset_states(self):
            if hasattr(self.base, "reset_states"):
                self.base.reset_states()
            self._mean.reset_states()
    # ---------------------------------------------------


    def compile_model(self, model):
        model: ln5.LeNet5_Like = model
        if self.args["opt"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(momentum=0.95, nesterov=True)
        elif self.args["opt"] == "adam":
            optimizer = tf.keras.optimizers.Adam(amsgrad=True)
        else:
            raise Exception(f"Unkown optimizer chosen \"" + self.args["opt"] + "\"")

        tf_top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top 1")
        non_strict = mq.LossesAndMetrics.create_top_k_accuarcy_fixed("fixed top 1", 1, is_strict=False)

        # Tolerance-based strict (tune tol; try 1e-3 if outputs are softmax probabilities)
        strict_tol = strict_top1_with_tolerance(tol=1e-4, name="fixed_(strict)_top_1")

        metrics = [tf_top1, non_strict, strict_tol]
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
        model.build([None] + self.INPUT_SHAPE)
        model.summary()

    def get_model(self, use_bnorm):
        model = ln5.LeNet5_Like(self.INPUT_SHAPE, [self.NUM_CLASSES], [self.NUM_CLASSES])
        return model

    def get_epochs_and_lrs(self, base_lr):
        if self.args["train-type"] == "float":
            EPOCHS_LRS = [(1, base_lr*10), (40, base_lr), (15, base_lr/10), (10, base_lr/100)]
        elif self.args["train-type"] == "fixed":
            EPOCHS_LRS = [(1, base_lr), (19, base_lr), (20,base_lr/10), (10, base_lr/100)]
        elif self.args["train-type"] == "lns":
            # Train for 40 epochs at constant LR
            EPOCHS_LRS = [(40, base_lr)]
        elif self.args["train-type"] == "adder":
            EPOCHS_LRS = [(10, base_lr*10), (10, base_lr), (10, base_lr/10), (10, base_lr/100)]
        else:
            raise Exception("UNKNOWN train type " + str(self.args["train-type"]))
        return EPOCHS_LRS

    def get_dataset(self, batch_size, cache_dataset):
        (train_dataset, test_dataset), dataset_info = tfds.load(
            name="mnist", shuffle_files=True, as_supervised=True, split=['train', 'test'], with_info=True
        )
        train_dataset = mq.DatasetUtilities.prepare_dataset(
            train_dataset, self.NUM_CLASSES, batch_size, dataset_info.splits['train'].num_examples, cache=True
        )
        test_dataset = mq.DatasetUtilities.prepare_dataset(test_dataset, self.NUM_CLASSES, batch_size, cache=True)
        return train_dataset, test_dataset


training = TrainLeNet()
training.parse_arguments()
training.train()

model = training.model
model: ln5.LeNet5_Like = model

quantized_conv1_weights = model.conv1.f()
unquantized_conv1_weights = model.conv1.f.var
quantized_dense1_weights = model.dense1.w()

print(np.unique(np.reshape(quantized_conv1_weights, -1)))

# === Reporting block: TF accuracy, Non-strict & Strict accuracy, and sfix settings ===
# Evaluate to get final metrics and loss
train_dataset, test_dataset = training.get_dataset(training.batch_size, cache_dataset=True)

# Training loss
train_eval = training.model.evaluate(train_dataset, verbose=0)
train_loss = train_eval[0]

# Validation (test) loss
test_eval = training.model.evaluate(test_dataset, verbose=0)
val_loss = test_eval[0]
counter = 0

for cur_batch in test_dataset.as_numpy_iterator():
    first_image = cur_batch[0][0:1, :, :, :]
    first_gt = cur_batch[1][0:1, :]
    print('image?', tf.shape(first_image))
    print('gt?', tf.shape(first_gt))

    tf.print("IT STARTS ")

    tmp = first_image
    tmp = model.conv1(tmp)
    tmp_weights = model.conv1.f()
    tmp_in = model.conv1.quant_in(tmp)
    print('conv1 weights', np.size(np.unique(np.reshape(tmp_weights, -1))), np.unique(np.reshape(tmp_weights, -1)))
    print('conv1 output', np.size(np.unique(np.reshape(tmp_in, -1))), np.unique(np.reshape(tmp_in, -1)))
    tmp = model.pool1(tmp)
    print('pool1 output', np.size(np.unique(np.reshape(tmp, -1))), np.unique(np.reshape(tmp, -1)))
    tmp = model.conv2(tmp)
    tmp_weights = model.conv2.f()
    tmp_in = model.conv2.quant_in(tmp)
    print('conv2 weights', np.size(np.unique(np.reshape(tmp_weights, -1))), np.unique(np.reshape(tmp_weights, -1)))
    print('conv2 output', np.size(np.unique(np.reshape(tmp_in, -1))), np.unique(np.reshape(tmp_in, -1)))
    tmp = model.pool2(tmp)
    print('pool2 output', np.size(np.unique(np.reshape(tmp, -1))), np.unique(np.reshape(tmp, -1)))
    tmp = model.flatten(tmp)
    tmp = model.dense1(tmp)
    tmp_weights = model.dense1.w()
    tmp_in = model.dense1.quant_in(tmp)
    print('dense weights', np.size(np.unique(np.reshape(tmp_weights, -1))), np.unique(np.reshape(tmp_weights, -1)))
    print('dense output', np.size(np.unique(np.reshape(tmp_in, -1))), np.unique(np.reshape(tmp_in, -1)))
    tf.print("model output", tmp, tf.argmax(tmp), summarize=-1)
    tf.print("model gt", first_gt, tf.argmax(first_gt), summarize=-1)
    tf.print("metrics", model.evaluate(first_image, first_gt, verbose=0))

    counter += 1
    if counter > 100:
        exit(3434)



#metrics_map = {m for m in training.model.metrics}
# metric_names = [m.name if hasattr(m, "name") else str(m) for m in training.model.metrics]
#metrics_map = {name: val for name, val in zip(metric_names, test_eval[1:])}  # skip loss at index 0
metrics_map = {
    "loss": test_eval[0],
    "top 1": test_eval[1],
    "fixed top 1": test_eval[2],
    "fixed_(strict)_top_1": test_eval[3],
}
print("test_eval", test_eval)

# Be extra safe if any metric is NaN
def _finite(x):
    try:
        return float(x) if np.isfinite(x) else None
    except Exception:
        return None

print("\n===== Final Metrics =====")
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"TensorFlow (top 1): {_finite(metrics_map.get('top 1', float('nan'))):.4f}")
print(f"Non-Strict Fixed (fixed top 1): {_finite(metrics_map.get('fixed top 1', float('nan'))):.4f}")
strict_tol = _finite(metrics_map.get('fixed_(strict)_top_1', float('nan')))
print(f"Strict (tolerance) Top-1 (fixed_(strict)_top_1): {strict_tol:.4f}")
print("=========================\n")

def _print_internal_sfix(label, q):
    try:
        internal_q = getattr(q, "internal_quantizer", None)
        if internal_q is not None and hasattr(internal_q, "get_sfix_settings"):
            print(f"{label} get_sfix_settings():", internal_q.get_sfix_settings())
            return True
    except Exception as e:
        print(f"{label} get_sfix_settings(): N/A ({e})")
    return False

printed_any = False
if hasattr(model, "conv1") and hasattr(model.conv1, "f") and hasattr(model.conv1.f, "quant_out"):
    printed_any |= _print_internal_sfix("conv1/f", model.conv1.f.quant_out)
if hasattr(model, "conv2") and hasattr(model.conv2, "f") and hasattr(model.conv2.f, "quant_out"):
    printed_any |= _print_internal_sfix("conv2/f", model.conv2.f.quant_out)
if hasattr(model, "dense1") and hasattr(model.dense1, "w") and hasattr(model.dense1.w, "quant_out"):
    printed_any |= _print_internal_sfix("dense1/w", model.dense1.w.quant_out)

if not printed_any:
    try:
        internal_quant = mq.FlexPointQuantizer("internal_fallback", 8)
        print("fallback get_sfix_settings():", internal_quant.get_sfix_settings())
    except Exception as e:
        print("fallback get_sfix_settings(): N/A", e)
# === End reporting block ===