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
# -tt fixed -qd layer-wise -b-bits 10 -w-bits 8 -a-bits 10 -desc test5 --checkpoint "data/training/LeNetLike/float/test5/model_best_q.npz" -no-skip


class TrainLeNet(QTraining):
    def __init__(self, INPUT_SHAPE=[28, 28, 1], NUM_CLASSES=10, batch_size=256):
        super().__init__("LeNetLike", INPUT_SHAPE, NUM_CLASSES, batch_size)

    def get_top1_float(self) -> float:
        return 0.9920

    def load_inital_model(self, model):
        pass

    def apply_ptq(self, model):
        return # No post training quantization here

    def quantize_model(self, model, selected_train_type, weights_bits_total, bias_bits_total,
                       activation_bits_total, adder, std_keep_factor, quantize_actication) -> None:
        args = self.args
        print("SELECTED selected_train_type", selected_train_type, f"and adder is \"{adder}\"")
        def get_quantizer(name, i="all", j="all", no_chan=False):
            suffix =  "" if no_chan else f"per_chan_{i}_{j}_f"
            if selected_train_type == "fixed":
                return mq.FlexPointQuantizer(name + suffix, weights_bits_total)
            if selected_train_type == "lns":
                raise Exception("Not implemented!")
            return mq.AddQuantizer(name + suffix, str(adder) + "Add" + str(weights_bits_total))

        quant_depth = args["quant-depth"]
        def quant_Conv(conv, conv_name, filters, input_channels, use_adder, special=False):
            if args["no-weight"] != True:
                if quant_depth == "kernel-wise":
                    conv.f.quant_out = mq.PerKernelQuantizer(f"{conv_name}_kw_f", get_quantizer, 3, filters, 2, input_channels)
                elif quant_depth == "channel-wise":
                    conv.f.quant_out = mq.PerChannelQuantizer(f"{conv_name}_cw_f", get_quantizer, 3, filters)
                elif quant_depth == "layer-wise":
                    conv.f.quant_out = get_quantizer(f"{conv_name}_f", no_chan=True)
                else:
                    raise Exception(f"Unkwon quant_depth \"{quant_depth}\"")
            if args["no-bias"] != True:
                conv.b.quant_out = mq.FlexPointQuantizer(f"{conv_name}_b", bias_bits_total)
            if args["no-activation"] != True:
                conv.quant_in = mq.FlexPointQuantizer(f"{conv_name}_out", activation_bits_total)

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
            if args["no-activation"] != True:
                dense.quant_in = mq.FlexPointQuantizer(f"{dense_name}_out", activation_bits_total)

        if selected_train_type == "fixed" or selected_train_type == "adder"or selected_train_type == "lns":
            use_adder = selected_train_type == "adder"
            quant_Conv(model.conv1, "QuantConv1", 8, 1, use_adder)
            quant_Conv(model.conv2, "QuantConv2", 16, 8, use_adder)
            quant_Dense(model.dense1, "QuantDense1", 10, use_adder)
        elif selected_train_type == "float":
            pass
        else:
            raise Exception("UNKNOWN train type!")

    def compile_model(self, model):
        model: ln5.LeNet5_Like = model
        if self.args["opt"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(momentum=0.95, nesterov=True)
        elif self.args["opt"] == "adam":
            optimizer = tf.keras.optimizers.Adam(amsgrad=True)
        else:
            raise Exception(f"Unkown optimizer chosen \"" + self.args["opt"] + "\"")
        metrics = [tf.keras.metrics.TopKCategoricalAccuracy(k=1, name="top 1"), mq.LossesAndMetrics.create_top_k_accuarcy_fixed("fixed top 1", 1, is_strict=False), mq.LossesAndMetrics.create_top_k_accuarcy_fixed("fixed (strict) top 1", 1, is_strict=True)]
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics) # mq.LossesAndMetrics.create_categorical_crossentropy_loss("loss")
        model.build([None] + self.INPUT_SHAPE) # 'categorical_crossentropy'  mq.LossesAndMetrics.create_categorical_crossentropy_loss_strict("loss") mq.LossesAndMetrics.CustomMSE()
        model.summary()

    def get_model(self, use_bnorm):
        model = ln5.LeNet5_Like(self.INPUT_SHAPE, [self.NUM_CLASSES], [self.NUM_CLASSES])
        return model

    def get_epochs_and_lrs(self, base_lr):
        if self.args["train-type"] == "float":
            EPOCHS_LRS = [(1, base_lr*10), (45, base_lr), (15, base_lr/10), (10, base_lr/100)]
        elif self.args["train-type"] == "fixed":
            EPOCHS_LRS = [(1, base_lr), (19, base_lr), (20,base_lr/10), (10, base_lr/100)]
        elif self.args["train-type"] == "adder":
            EPOCHS_LRS = [(20, base_lr), (60,base_lr/10), (30, base_lr/100)] # [(10, lr_base/100), (30, lr_base/10), (30, lr_base/100), (10, lr_base/1000)] # (5, lr_base/100), (5, lr_base/10),
            EPOCHS_LRS = [(10, base_lr*10), (10, base_lr), (10, base_lr/10), (10, base_lr/100), ]
        else:
            raise Exception("UNKNOWN train type " + str(self.args["train-type"]))
        return EPOCHS_LRS


    def get_dataset(self, batch_size, cache_dataset):
        (train_dataset, test_dataset), dataset_info = tfds.load(name="mnist", shuffle_files=True,
                                                                as_supervised=True, split=['train', 'test'], with_info=True)

        train_dataset = mq.DatasetUtilities.prepare_dataset(
            train_dataset, self.NUM_CLASSES, batch_size, dataset_info.splits['train'].num_examples,
            cache=True)  # .take(20)
        test_dataset = mq.DatasetUtilities.prepare_dataset(test_dataset, self.NUM_CLASSES, batch_size, cache=True)
        return train_dataset, test_dataset


training = TrainLeNet()
training.parse_arguments()
training.train()