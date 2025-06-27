# -*- coding: utf-8 -*-
import os
import sys

script_folder_path = os.path.dirname(os.path.realpath(__file__))
mquat_path = os.path.split(os.path.split(script_folder_path)[0])[0]
sys.path.append(mquat_path)

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
import mquat as mq
import mquat.Datasets.CiFar10 as cifar10
import datetime



def start(model_class, quantize_method, LR_FACTOR, total_bits, KEEP_FACTOR, with_activation, EPOCHS, starting_checkpoint):

    train_record_path = None # None -> default tfds path like C:\Users\USERNAME\tensorflow_datasets
    test_record_path = train_record_path

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    # 0.1 false 10 12 8 0 "ResNet18 q(9)_float.npz"


    starting_checkpoint_fused = os.path.basename(starting_checkpoint)[0:-4] + "fused"

    print("LR_FACTOR", LR_FACTOR, "total_bits", total_bits, "EPOCHS", EPOCHS)
    print("sys.argv", sys.argv)
    LEARN_RATE = 0.0001 * LR_FACTOR

    WEIGHT_DECAY = 0  # 1e-4
    TRAIN_BATCH_SIZE = 128  # 32
    TEST_BATCH_SIZE = 128  # 32

    train_dataset = cifar10.get_train_ds(TRAIN_BATCH_SIZE, train_record_path).take(10)
    test_dataset = cifar10.get_test_ds(TEST_BATCH_SIZE, test_record_path).take(10)

    if ".npz" in starting_checkpoint:
        checkpoint = starting_checkpoint
    else:
        checkpoint = "checkpoint_latest.npz"

    FLEX_SAMPLE_SIZE = 4
    tf.keras.backend.clear_session()
    model_quant_desc = "q(" + str(total_bits)
    if with_activation:
        model_quant_desc += ",act"
    model_quant_desc += ")"
    print("quantisize with", model_quant_desc, flush=True)

    model = model_class([cifar10.IMAGE_CROP_SIZE, cifar10.IMAGE_CROP_SIZE, 3], [10], [10], WEIGHT_DECAY, tainable_bias=True, do_batch_norm=False)
    model_quant_desc = model.name + "_" + model_quant_desc
    #model.removeAllBatchNormLayers() # todo add me back just like the use_batchnorm=False in model creation

    #todo add with_bias to the sys.argv and add with_weights, with_mul and with_sum to the whole mix
    quantize_method(model, total_bits, FLEX_SAMPLE_SIZE, KEEP_FACTOR, with_activation, with_bias=False, debug=False)

    optimizer = tf.keras.optimizers.SGD(lr=LEARN_RATE, momentum=0.94, nesterov=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=0.9, beta_2=0.999, amsgrad=True, decay=1e-6)
    metrics = ['accuracy', mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 1 fixed", 1),
               mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 5 fixed", 5)]
    model.compile(loss=mq.LossesAndMetrics.categorical_crossentropy_with_epsilon,
                  optimizer=optimizer, metrics=metrics)
    model.build([None, cifar10.IMAGE_CROP_SIZE, cifar10.IMAGE_CROP_SIZE, 3])
    model.summary()
    print("loading checkpoint ", checkpoint)
    #model.load_pretrained_weights()
    model.loadVariablesNPZ(checkpoint)
    all_quantizer = model.getQuantizers()
    for quant in all_quantizer:
        quant.reset()

    # if os.path.exists("opt_" + checkpoint):
    #     model.loadOptimizerNPZ("opt_" + checkpoint)
    print("pretrained model created!")

    print("evaluation before training")
    evaluation_results = model.evaluate(test_dataset)
    print("evaluation before training done", evaluation_results)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
    #                               patience=5, min_lr=LEARN_RATE/1000)

    class LocalCallbacks(callbacks.Callback):
        def __init__(self, ptq_acc):
            self.best_acc = ptq_acc
            self.save_model_as_best()

        def save_model_as_best(self):
            model.saveVariablesNPZ(f"{model_quant_desc}", quantisize=False)
            model.saveOptimizerNPZ(f"opt_{model_quant_desc}")
            model.saveVariablesNPZ(f"{model_quant_desc}_quan", quantisize=True)
            model.saveVariablesNPZ("checkpoint_latest", quantisize=False)
            model.saveOptimizerNPZ("opt_checkpoint_latest")
            model.saveVariablesNPZ("checkpoint_latest_quan", quantisize=True)

        def on_epoch_end(self, epoch, logs={}):
            try:
                if os.path.exists("lr_callback.txt"):
                    f = open("lr_callback.txt", "r")
                    text = f.readline()
                    lr_div = float(text)
                    old_lr = model.optimizer.lr.read_value()
                    f.close()
                    os.remove("lr_callback.txt")
                    model.optimizer.lr.assign(old_lr * lr_div)
                    print("old lr", old_lr, "new lr", model.optimizer.lr.read_value())
            except Exception as e:
                print("Setting lr failed", e)
            if self.best_acc < logs["val_top 1 fixed"]:
                self.best_acc = logs["val_top 1 fixed"]
                print()
                print("found a better one for", model_quant_desc, " with ", logs["val_top 1 fixed"], logs["val_top 5 fixed"], flush=True)
                self.save_model_as_best()


    csv_logger = callbacks.CSVLogger('log.csv', append=True, separator=';')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("lr",model.optimizer.lr, flush=True)
    model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, verbose=1, use_multiprocessing=True,
              validation_freq=1, callbacks=[LocalCallbacks(evaluation_results[1]), csv_logger, tensorboard_callback])

    print("done training to quantisize with", model_quant_desc, flush=True)
