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

def start(model_class, quantize_method, LR_FACTOR, total_bits, with_activation, EPOCHS, starting_checkpoint,
          KEEP_FACTOR, batch_size=32, adder=None, fused_bnorm=True, keep_quantizers=False, model_quant_desc_prefix="",
          squant_model=None, model_args={}, tainable_bias=True):
    train_record_path = None # None -> default tfds path like C:\Users\USERNAME\tensorflow_datasets
    test_record_path = train_record_path
    # if not keep_quantizers:
    #     batch_size=4
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # 0.1 false 10 12 8 0 "ResNet18 q(9)_float.npz"

    print("LR_FACTOR, total_bits, with_activation, EPOCHS, starting_checkpoint, "
          "KEEP_FACTOR, batch_size, adder, fused_bnorm, keep_quantizers, model_quant_desc_prefix",
          LR_FACTOR, total_bits, with_activation, EPOCHS, starting_checkpoint,
          KEEP_FACTOR, batch_size, adder, fused_bnorm, keep_quantizers, model_quant_desc_prefix)


    tf.print("LR_FACTOR", LR_FACTOR, "total_bits", total_bits, "EPOCHS", EPOCHS)
    tf.print("sys.argv", sys.argv)
    LEARN_RATE = 0.0001 * LR_FACTOR

    WEIGHT_DECAY = 0  # 1e-4
    TRAIN_BATCH_SIZE = batch_size  # 32
    TEST_BATCH_SIZE = batch_size  # 32

    train_dataset = img1k.get_train_ds(TRAIN_BATCH_SIZE, train_record_path)
    test_dataset = img1k.get_test_ds(TEST_BATCH_SIZE, test_record_path)

    if ".npz" in starting_checkpoint:
        checkpoint = starting_checkpoint
    else:
        checkpoint = "checkpoint_latest.npz"

    FLEX_SAMPLE_SIZE = 3
    tf.keras.backend.clear_session()
    model_quant_desc = model_quant_desc_prefix + "q("
    if adder != None:
        model_quant_desc += str(adder)+"Add_"
    model_quant_desc += str(total_bits)
    if with_activation:
        model_quant_desc += ",act"
    model_quant_desc += ")"
    tf.print("quantisize with", model_quant_desc)

    if fused_bnorm:
        model = model_class([img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3], [1000], [1000], WEIGHT_DECAY, **model_args, tainable_bias=True, do_batch_norm=False)
        model.removeAllBatchNormLayers()
    else:
        model = model_class([img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3], [1000], [1000], WEIGHT_DECAY, **model_args, tainable_bias=tainable_bias, do_batch_norm=True)
    model_quant_desc = model.name + "_" + model_quant_desc
    tf.print("model_quant_desc", model_quant_desc)

    #todo add with_bias to the sys.argv and add with_weights, with_mul and with_sum to the whole mix
    quantize_method(model=model, total_bits=total_bits, FLEX_SAMPLE_SIZE=FLEX_SAMPLE_SIZE, with_activation=with_activation, KEEP_FACTOR=KEEP_FACTOR, with_bias=True, adder=adder, debug=False)

    optimizer = tf.keras.optimizers.SGD(lr=LEARN_RATE, momentum=0.94, nesterov=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE, beta_1=0.9, beta_2=0.999, amsgrad=True, decay=1e-6)
    metrics = ['accuracy', mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 1 fixed", 1),
               mq.LossesAndMetrics.create_top_k_accuarcy_fixed("top 5 fixed", 5)]
    model.compile(loss=mq.LossesAndMetrics.categorical_crossentropy_with_epsilon,
                  optimizer=optimizer, metrics=metrics)
    model.build([None, img1k.IMAGE_CROP_SIZE, img1k.IMAGE_CROP_SIZE, 3])
    model.summary()
    model.load_pretrained_weights()

    if checkpoint != ".npz":
        tf.print("loading checkpoint ", checkpoint)
        model.loadVariablesNPZ(checkpoint)
        if os.path.exists("opt_" + checkpoint):
            try:
                model.loadOptimizerNPZ("opt_" + checkpoint)
            except:
                pass
        if not keep_quantizers:
            tf.print("RESETTING ALL QUANTIZERS!!!")
            all_quantizer = model.getQuantizers()
            for quant in all_quantizer:
                quant.reset()
        else:
            tf.print("KEEPING OLD QUANTIZERS!!")
    else:
        print("LOADING PRE TRAINED WEIGHTS FROM TF2CV")
        model.load_pretrained_weights()
    tf.print("pretrained model created!")

    tf.print("evaluation before training")

    # evaluation_results = model.evaluate(test_dataset)
    # tf.print("evaluation before training done", evaluation_results)
    if squant_model is not None:
        tf.print()
        tf.print("Apply squant")
        #model.saveVariablesNPZ("tmp_model_before_squant")

        squant_model(model)

        model.saveVariablesNPZ("tmp_model_after_squant2")
        tf.print("Done applying")
    evaluation_results = model.evaluate(test_dataset)
    tf.print("evaluation before training done", evaluation_results)


    # if not keep_quantizers:
    #     model.saveVariablesNPZ(f"last_ptq", quantisize=False)
    #     model.saveVariablesNPZ(f"last_ptq_quan", quantisize=True)
    #     exit(0)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
    #                               patience=5, min_lr=LEARN_RATE/1000)

    if EPOCHS != 0:
        class LocalCallbacks(callbacks.Callback):
            def __init__(self, ptq_acc):
                self.best_acc = 0.0
                self.best_acc5 = 0.0
                self.best_epoch = 0

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
                    self.best_acc5 = logs["val_top 5 fixed"]
                    self.best_epoch = epoch
                    print()
                    print("found a better one for", model_quant_desc, " with ", logs["val_top 1 fixed"], logs["val_top 5 fixed"], flush=True)
                    self.save_model_as_best()

            def on_train_end(self, logs={}):
                print("best result for", model_quant_desc, "is", self.best_acc, self.best_acc5, self.best_epoch, flush=True)


        csv_logger = callbacks.CSVLogger('log.csv', append=True, separator=';')

        print("lr",model.optimizer.lr, flush=True)
        model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, verbose=1, use_multiprocessing=True,
                  validation_freq=1, callbacks=[LocalCallbacks(0.0), csv_logger])
        print("done training to quantisize with", model_quant_desc, flush=True)
    else:
        return evaluation_results