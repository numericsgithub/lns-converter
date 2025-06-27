# -*- coding: utf-8 -*-
import os
import sys

script_folder_path = os.path.dirname(os.path.realpath(__file__))
mquat_path = os.path.split(os.path.split(script_folder_path)[0])[0]
sys.path.append(mquat_path)

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
import mquat as mq
from mquat.Datasets import ImageNet1K as img1k

def start(model_class, quantize_method, LR_FACTOR, total_bits, with_activation, EPOCHS, starting_checkpoint, KEEP_FACTOR, with_bias=True, adder=None, fused_bnorm=True):

    train_record_path = None # None -> default tfds path like C:\Users\USERNAME\tensorflow_datasets
    test_record_path = train_record_path

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # 0.1 false 10 12 8 0 "ResNet18 q(9)_float.npz"


    print("LR_FACTOR", LR_FACTOR, "total_bits", total_bits, "EPOCHS", EPOCHS)
    print("sys.argv", sys.argv)
    LEARN_RATE = 0.0001 * LR_FACTOR

    WEIGHT_DECAY = 0  # 1e-4
    TRAIN_BATCH_SIZE = 32  # 32
    TEST_BATCH_SIZE = 32  # 32

    train_dataset = img1k.get_train_ds(TRAIN_BATCH_SIZE, train_record_path)
    test_dataset = img1k.get_test_ds(TEST_BATCH_SIZE, test_record_path)

    if ".npz" in starting_checkpoint:
        checkpoint = starting_checkpoint
    else:
        checkpoint = "checkpoint_latest.npz"

    FLEX_SAMPLE_SIZE = 3
    tf.keras.backend.clear_session()
    model_quant_desc = "q("
    if adder != None:
        model_quant_desc += str(adder)+"Add "
    model_quant_desc += str(total_bits)
    if with_activation:
        model_quant_desc += ",act"
    model_quant_desc += ")"
    print("quantisize with", model_quant_desc, flush=True)

    if fused_bnorm:
        model = model_class([img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3], [1000], [1000], WEIGHT_DECAY, tainable_bias=True, do_batch_norm=False)
        model.removeAllBatchNormLayers()
    else:
        model = model_class([img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3], [1000], [1000], WEIGHT_DECAY, tainable_bias=True, do_batch_norm=True)
    model_quant_desc = model.name + "_" + model_quant_desc

    #todo add with_bias to the sys.argv and add with_weights, with_mul and with_sum to the whole mix
    quantize_method(model=model, total_bits=total_bits, FLEX_SAMPLE_SIZE=FLEX_SAMPLE_SIZE, with_activation=with_activation, KEEP_FACTOR=KEEP_FACTOR, with_bias=with_bias, debug=False)

    optimizer = tf.keras.optimizers.SGD(lr=LEARN_RATE, momentum=0.94, nesterov=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=0.9, beta_2=0.999, amsgrad=True, decay=1e-6)
    metrics = ['accuracy', mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 1 fixed", 1),
               mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 5 fixed", 5)]
    model.compile(loss=mq.LossesAndMetrics.categorical_crossentropy_with_epsilon,
                  optimizer=optimizer, metrics=metrics)
    model.build([None, img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3])
    model.summary()
    print("loading checkpoint ", checkpoint)
    model.loadVariablesNPZ(checkpoint)
    if os.path.exists("opt_" + checkpoint):
        try:
            model.loadOptimizerNPZ("opt_" + checkpoint)
        except:
            pass
    # all_quantizer = model.getQuantizers()
    # for quant in all_quantizer:
    #     quant.reset()
    print("pretrained model created!")

    evaluation_results = model.evaluate(test_dataset)
    print("evaluation done", evaluation_results, sys.argv)
    val_path = "validated"
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    val_filename = os.path.basename(checkpoint)
    val_filename = os.path.splitext(val_filename)[0]
    model.saveVariablesNPZ(f"{val_path}/{val_filename}_quan", quantisize=True)
    model.saveVariablesNPZ(f"{val_path}/{val_filename}", quantisize=False)
    with open(f"{val_path}/{val_filename}_result.txt", "w") as text_file:
        text_file.write(str(evaluation_results))