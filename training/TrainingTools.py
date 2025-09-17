import os
import shutil
import sys
import json
from typing import List, Tuple

import numpy as np
from keras.callbacks import ReduceLROnPlateau

#sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import mquat as mq  # noqa: E402
from tensorflow import keras
import datetime  # noqa: E402
import tensorflow.keras.callbacks as callbacks
import tensorflow as tf
from argparse import ArgumentParser
import time
import utils.SimpleTelebotReport as trep
from mquat.Datasets import ImageNet1K as img1k
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from utils import SplitLogger, DataPaths
from utils.DataPaths import datpat
from utils.LRFinder import LRFinder, find_lr
from utils.TrainingCallbacks import TrainingCallbacks, TrainingCallbacksData

def get_dataset_IMAGENET(NUM_CLASSES, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, data_path=None, cache_dataset=False, fast_mode=True) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
    train_dataset = img1k.get_train_ds(TRAIN_BATCH_SIZE, train_record_path=data_path, cache=cache_dataset, fast_mode=fast_mode)
    test_dataset = img1k.get_test_ds(TEST_BATCH_SIZE, test_record_path=data_path, cache=False)
    return train_dataset, test_dataset

def load_inital_model(model, args):
    model:mq.QNetBaseModel = model
    model.load_pretrained_weights()
    # model.saveVariablesNPZ("data/training/pretrained/pretrained_resnet18_with_bn.npz")
    # model.saveVariablesNPZ("data/training/pretrained/pretrained_resnet18_without_bn.npz", fuse_batch_norm=True)
    # #model.loadVariablesNPZ("data/training/pretrained/pretrained_resnet18_with_bn.npz")

def create_inital_npz(model, args):
    model:mq.QNetBaseModel = model
    model.load_pretrained_weights()
    print(f"data/training/pretrained/pretrained_{args['model-name']}_with_bn.npz")
    model.saveVariablesNPZ(f"data/training/pretrained/pretrained_{args['model-name']}_with_bn.npz")
    model.saveVariablesNPZ(f"data/training/pretrained/pretrained_{args['model-name']}_without_bn.npz", fuse_batch_norm=True)
    #model.loadVariablesNPZ("pretrained_resnet18_with_bn.npz")


def func_ptq(model, selected_train_type, weights_bits_total, bias_bits_total,
             activation_bits_total, adder, std_keep_factor, model_load_path, args):
    pass

class QTraining:

    def __init__(self, model_name, INPUT_SHAPE, NUM_CLASSES, batch_size):
        self.model_name = model_name
        self.INPUT_SHAPE = INPUT_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.batch_size:int = batch_size
        self.args:dict = {}
        self.model = None

    def apply_ptq(self, model) -> None:
        pass

    def get_top1_float(self) -> float:
        pass

    def load_inital_model(self, model) -> None:
        pass

    def get_epochs_and_lrs(self, base_lr) -> List[Tuple[int, float]]:
        pass

    def compile_model(self, model) -> None:
        pass

    def quantize_model(self, model, selected_train_type, weights_bits_total, bias_bits_total,
                        activation_bits_total, adder, std_keep_factor, quantize_actication) -> None:
        pass

    def get_model(self, use_bnorm) -> mq.QNetBaseModel:
        pass

    def get_dataset(self, batch_size, cache_dataset) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        return get_dataset_IMAGENET(self.NUM_CLASSES, batch_size, batch_size, self.args["dataset_path"], self.args["cache-dataset"] and cache_dataset, self.args["hard-train"])

    def get_train_run_summary(self):
        def qbits_str(bits):
            return "N" if bits is None else str(bits)
        wbaA = [qbits_str(self.args["weight-bits"]), qbits_str(self.args["bias-bits"]), qbits_str(self.args["activation-bits"]), qbits_str(self.args["adder-type"])]
        wbaA = ",".join(wbaA)
        desc = str(self.args["desc"])
        lr_base = self.args["lr_base"]
        mobnet_alpha = self.args["mobnet_alpha"]
        bsize = self.batch_size_big
        use_bnorm = self.args["use-bnorm"]
        extra_info = ""
        if lr_base is not None:
            extra_info += "lr_base(" + str(lr_base) + ")"
        if use_bnorm is not None:
            extra_info += " use_bnorm(" + str(use_bnorm) + ")"
        if bsize is not None:
            extra_info += " bsize(" + str(bsize) + ")"
        if mobnet_alpha is not None:
            extra_info += " mobnet_alpha(" + str(mobnet_alpha) + ")"
        summary = f"{self.model_name}({desc}): wbaA({wbaA}) " + extra_info
        return summary

    def train(self):
        args = self.args
        model_name = self.model_name
        INPUT_SHAPE = self.INPUT_SHAPE
        NUM_CLASSES = self.NUM_CLASSES
        # batch_size = self.batch_size

        train_type = args["train-type"]
        quant_depth = args["quant-depth"]
        model_load_path = args["checkpoint"]
        bias_bits_total = args["bias-bits"]
        weights_bits_total = args["weight-bits"]
        activation_bits_total = args["activation-bits"]
        adder = args["adder-type"]
        general_desc = args["desc"]
        dataset_path = args["dataset_path"]
        time_start = time.time()
        # lr_base = args["lr_base"]
        no_quant_reset = args["no-quant-reset"]
        cache_dataset = args["cache-dataset"]
        special_action = args["action"]
        quantize_actication = True
        
        try:
            model_base_path, MODEL_SAVEPATH, MODEL_SAVEPATH_Q, MODEL_SAVEPATH_BEST, MODEL_SAVEPATH_Q_BEST, MODEL_SAVEPATH_Q_FUSEB = DataPaths.get_model_paths(args)
            print("model base path", model_base_path)
            os.makedirs(model_base_path.format(""), exist_ok=True)

            SplitLogger.duplicate_logs(model_base_path)

            if os.path.exists(MODEL_SAVEPATH):
                print(f"Already done! Skipping! Because {MODEL_SAVEPATH} already exists")
                if args["no-skip"] == True:
                    print("SKIPPING PREVENTED DUE TO no-skip FLAG!")
                else:
                    exit(0)

            train_dataset_small, _ = self.get_dataset(32, False)

            std_keep_factor = 0.0
            use_bnorm = special_action == "create_npz"
            if self.args["use-bnorm"]:
                use_bnorm = True
            model = self.get_model(use_bnorm)

            self.quantize_model(model, train_type, weights_bits_total, bias_bits_total, activation_bits_total, adder, std_keep_factor, quantize_actication)

            self.compile_model(model)

            if special_action == "create_npz":
                create_inital_npz(model, args)
                # model.evaluate(test_dataset, verbose=2)
                exit(0)


            if model_load_path != None:
                model.loadVariablesNPZ(model_load_path, try_load_quantizers=False) # loss: 0.0137 - top 1: 0.9960 - val_loss: 0.0302 - val_top 1: 0.9908
            else:
                if "tiny_imgnet" in model_name:
                    self.load_inital_model(model)
                elif model_name != "LeNetLike":
                    model.loadVariablesNPZ(f"data/training/pretrained/pretrained_{args['model-name']}_without_bn.npz", try_load_quantizers=False)

            self.model = model
            # all_quants = model.getQuantizers()
            # all_quants_2 = find_instances_of_my_class_a(model)
            # print("ALL QUANTIZERS", len(all_quants), [q.name for q in all_quants])
            # print("ALL QUANTIZERS2", len(all_quants_2), [q.name for q in all_quants_2])
            # exit(0)

            if no_quant_reset and model_load_path != None:
                quant_vars_filepath = os.path.splitext(model_load_path)[0] + "_qvars.npz"
                if os.path.exists(quant_vars_filepath):
                    print(quant_vars_filepath, "was found. Now loading quantizer variables")
                    model.loadQuantVariablesNPZ(quant_vars_filepath)
                else:
                    raise Exception("Cannot load quantizers. So not resetting quantizers does not make sense!")
            else:
                for q in model.getQuantizers():
                    q.reset()
            func_ptq(model, train_type, weights_bits_total, bias_bits_total, activation_bits_total, adder,
                     std_keep_factor, model_load_path, args)
            for v in model.getVariables():
                v: mq.Variable = v
                v()

            print("initialize the activation quantization")
            started = time.time()
            model.evaluate(train_dataset_small.take(10), verbose=2) # initialize the activation quantization
            print()
            print("TOOK: ", time.time() - started)
            print()
            if special_action == "quick_test_npz":
                exit(0)
            #train_dataset = train_dataset.cache("/run/determined/workdir/shared/datasets/img2012_train")
            train_dataset, test_dataset = self.get_dataset(self.batch_size_big, True)
            print("Evaluation on test dataset")
            evaluation_results = model.evaluate(test_dataset, verbose=2)
            print("Evaluation finished")

            if special_action == "test_npz":
                exit(0)

            print("\n\tTYPE", train_type, "\n\tmodel_load_path", model_load_path, "\n\tbias_bits_total", bias_bits_total,
                  "\n\tweights_bits_total", weights_bits_total, "\n\tactivation_bits_total", activation_bits_total, "\n\tadder", adder)
            with open(datpat("training/before_train_results.txt"), "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(evaluation_results)}\r\n")
            print(evaluation_results)

            trep.sendModelReport(model_name, args, "Started training with " + str(args))

            if args["lr_base"] is None:
                best_lr = find_lr(model, train_dataset, test_dataset, model_base_path, 0, True, self.args)
            else:
                best_lr = float(args["lr_base"])
                print("NO LR FINDING! LR WAS SET TO ", best_lr)

            EPOCHS_LRS = self.get_epochs_and_lrs(best_lr)# [(10, best_lr), (3, best_lr * 5), (12, best_lr), (3, best_lr * 2.5), (10, best_lr), (2, best_lr / 2.5)] # self.get_epochs_and_lrs(best_lr)

            # remove previous results
            if os.path.exists(MODEL_SAVEPATH):
                os.remove(MODEL_SAVEPATH)
            if os.path.exists(MODEL_SAVEPATH_Q):
                os.remove(MODEL_SAVEPATH_Q)

            training_callbacks_data = TrainingCallbacksData(model, model_name, args, self.get_top1_float())


            #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tensorboard_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1, profile_batch=(1,20)) # tensorboard --logdir tensorboard_logs
            for run_id, (epochs, lr) in enumerate(EPOCHS_LRS):
                if training_callbacks_data.early_stopped:
                    print("EARLY STOP DETECTED!")
                    continue
                model.optimizer.lr.assign(lr)

                reduce_lr_callback = ReduceLROnPlateau() # cooldown=2, # monitor="loss", mode="min", patience=5, factor=0.5, min_lr=1e-9
                my_callback = TrainingCallbacks(training_callbacks_data, self.get_train_run_summary())

                results: keras.callbacks.History = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, use_multiprocessing=True, verbose=1 if run_id == 0 else 2,
                          validation_freq=1, callbacks=[my_callback, reduce_lr_callback])
                training_callbacks_data.add_event_at_cur_epoch(f"L({training_callbacks_data.cur_best_model_for_training_epoch})")
                if os.path.exists(MODEL_SAVEPATH):
                    model.loadVariablesNPZ(MODEL_SAVEPATH)
                else:
                    if model_load_path != None:
                        print(f"Resetting to {model_load_path}")
                        trep.sendModelReport(model_name, args, f"Resetting to {model_load_path}")
                        model.loadVariablesNPZ(model_load_path, try_load_quantizers=False)  # loss: 0.0137 - top 1: 0.9960 - val_loss: 0.0302 - val_top 1: 0.9908
                    else:
                        self.load_inital_model(model)
                # old_lr = model.optimizer.lr.read_value()
                # model.optimizer.lr.assign(old_lr * 0.1)
            my_callback.you_are_done()

            extra_info = {"epochs_count": str(training_callbacks_data.epochs_counter), "time": str(time.time()-time_start), "best_epoch": str(training_callbacks_data.best_acc_epoch)}
            evaluation_results = model.evaluate(test_dataset, verbose=2)
            with open(datpat("training/after_train_results.txt"), "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(args)} | {str(evaluation_results)}\n")
            with open(datpat("training/after_train_best_results.txt"), "a") as f:
                f.write(f"{MODEL_SAVEPATH} | {model_load_path} | {str(args)} | {str(training_callbacks_data.best_acc_results)} | {str(extra_info)}\n")
                trep.sendModelReport(model_name, args, f"Best Result is {training_callbacks_data.best_acc_results.get('val_fixed_(strict)_top_1',training_callbacks_data.best_acc_results.get('val_fixed (strict) top 1'))} fp32 would be {self.get_top1_float()}")
            with open(model_base_path.format("extra_info.txt"), "w") as f:
                f.write(str(extra_info))
            print(evaluation_results)
            return model
        except Exception as ex:
            trep.sendModelReport(model_name, args, f"Fatal Error! {ex}")
            raise ex
        
        
    def parse_arguments(self):
        parser = ArgumentParser()
        parser.add_argument("-desc", dest="desc", default="",
                            help="Short description. A folder with the description as a name will contain all files generated")
        parser.add_argument("--checkpoint", dest="checkpoint",
                            help="Checkpoint file to load")
        parser.add_argument("-tt", "--train-type", choices=["float", "fixed", "adder", "lns"], dest="train-type",
                            required=True,
                            help="Set what kind of quantizers you want to choose. Either none at all, fixed or adder aware")

        parser.add_argument("-qd", "--quant-depth", choices=["layer-wise", "channel-wise", "kernel-wise"],
                            dest="quant-depth", default="layer-wise",
                            help="Set how to split up each quantization. Quantize each layer, channel or kernel with a seperate quantizer")
        parser.add_argument('-b-bits', '--bias-bits', type=int, choices=range(1, 21), dest="bias-bits")
        parser.add_argument('-w-bits', '--weight-bits', type=int, choices=range(1, 21), dest="weight-bits")
        parser.add_argument('-a-bits', '--activation-bits', type=int, choices=range(1, 21), dest="activation-bits")

        parser.add_argument('-no-b-quant', '--no-bias-quant', action='store_true', dest="no-bias")
        parser.add_argument('-no-w-quant', '--no-weight-quant', action='store_true', dest="no-weight")
        parser.add_argument('-no-a-quant', '--no-activation-quant', action='store_true', dest="no-activation")
        parser.add_argument('-no-skip', '--no-skip', action='store_true', dest="no-skip")
        parser.add_argument('-no-quant-reset', '--no-quant-reset', action='store_true', dest="no-quant-reset")

        parser.add_argument('-cache-ds', '--cache-ds', action='store_true', dest="cache-dataset")
        parser.add_argument('-hard-train', '--hard-train', action='store_false', dest="hard-train")

        parser.add_argument("--action", dest="action", help="Choose special actions")
        parser.add_argument("--gpu", dest="gpu", help="Choose specific gpu")
        parser.add_argument('-use-bnorm', '--use-bnorm', action='store_true', dest="use-bnorm")

        parser.add_argument('-adder-type', dest="adder-type")
        parser.add_argument("--dataset-path", dest="dataset_path", default=None,
                            help="Path to the tensorflow_datasets folder")
        parser.add_argument('-bsize', type=int, dest="bsize", default=None)
        parser.add_argument('--experimental-MUX-thr', type=float, dest="exp-MUX-thr", default=None)
        parser.add_argument('-lr', type=float, dest="lr_base", default=None)
        parser.add_argument('-mobnet-alpha', type=float, dest="mobnet_alpha", default=None)
        parser.add_argument('-opt', choices=["adam", "sgd"], dest="opt", required=False, default="adam",
                            help="Set optimizer")
        parser.add_argument("--lns-format", type=str, default="sfix", help="LNS format: sfix or ufix")
        parser.add_argument("--lns-lsb", type=int, default=-3, help="Least significant bit for LNS quantizer")
        parser.add_argument("--lns-msb", type=int, default=1, help="Most significant bit for LNS quantizer")
        _ = parser.parse_args()
        args = {}
        for arg in vars(_):
            args[arg] = getattr(_, arg)
        print(args)

        if args["train-type"] == "float":
            has_no_quant_settings = args["bias-bits"] == None and args["weight-bits"] == None and args[
                "activation-bits"] == None
            has_no_quant_settings = has_no_quant_settings and args["adder-type"] == None
            has_no_quant_settings = has_no_quant_settings and args["no-bias"] == False and args[
                "no-weight"] == False and args["no-activation"] == False
            if not has_no_quant_settings:
                raise Exception(
                    "Wrong Arguments! When training without quantization, quantization arguments are not allowed!")
        else:
            if args["bias-bits"] == None and args["no-bias"] == False:
                raise Exception(
                    "Quantization of the bias is not specified! Either set --no-b-quant flag or set bits via --b-bits 8")
            if args["weight-bits"] == None and args["no-weight"] == False:
                raise Exception(
                    "Quantization of the weight is not specified! Either set --no-w-quant flag or set bits via --w-bits 8")
            if args["activation-bits"] == None and args["no-activation"] == False:
                raise Exception(
                    "Quantization of the activation is not specified! Either set --no-a-quant flag or set bits via --a-bits 8")
            if args["quant-depth"] == None:
                raise Exception("Quantization depth is not specified! Set this via --quant-depth")
            if args["train-type"] == "adder":
                if args["adder-type"] == None:
                    raise Exception("Adder type is not specified! Set this via --adder-type")
            if args["train-type"] == "fixed":
                if args["adder-type"] != None:
                    raise Exception("Adder type is specified! But train type is \"fixed\"!")


        args["model-name"] = self.model_name
        if args["mobnet_alpha"] != None:
            args["model-name"] += "_" + str(args["mobnet_alpha"])
        self.model_name = args["model-name"]

        if args["bsize"] is None:
            print("BSIZE WAS NONE!")
            args["bsize"] = self.batch_size
        print("BSIZE OVERWRITE", args["bsize"])
        time.sleep(5)
        self.batch_size_big = args["bsize"]
        self.batch_size = args["bsize"]

        self.args = args
        return args
