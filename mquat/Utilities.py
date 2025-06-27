# -*- coding: utf-8 -*-

import os
import sys
from typing import List

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import random
import time
import json
import gc

import mquat as mq


# set all used random number generators to a given seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
        
def satuate_color(color, saturation):
    color = colorsys.rgb_to_hls(*color[:3])
    return colorsys.hls_to_rgb(color[0], 1 - saturation * (1 - color[1]), color[2])
    
# evaluate model for train and valid datasets
def evaluate_model(model, datasets_fn):
    train_data, valid_data = datasets_fn()
    train_metrics = model.evaluate(train_data, verbose=0, return_dict=True)
    train_metrics = { f"train_{name}": value for name, value in train_metrics.items() }
    valid_metrics = model.evaluate(valid_data, verbose=0, return_dict=True)
    valid_metrics = { f"valid_{name}": value for name, value in valid_metrics.items() }
    tmp_metrics = { **train_metrics, **valid_metrics }
    metrics = {}
    for name, value in tmp_metrics.items():
        if name.startswith("valid_"):
            if isinstance(model, mq.QNetMultiModel) and len(model.getSubModels()) == 1:
                name = name.replace("valid_", f"valid_{model.getSubModels()[0].name}_")
            metrics[name] = value
        else:
            if isinstance(model, mq.QNetMultiModel) and len(model.getSubModels()) == 1:
                name = name.replace("train_", f"train_{model.getSubModels()[0].name}_")
            metrics[name] = value
    # delete redundant total loss if MultiModel is present
    if isinstance(model, mq.QNetMultiModel):
        if "train_loss" in metrics: 
            del metrics["train_loss"]
        if "valid_loss" in metrics: 
            del metrics["valid_loss"]          
    return metrics

def fit_model_epoch(model, datasets_fn):
    train_data, valid_data = datasets_fn()
    tmp_metrics = model.fit(train_data, epochs=1, validation_data=valid_data, verbose=0).history
    metrics = {}
    for name, value in tmp_metrics.items():
        if name.startswith("val_"):
            if isinstance(model, mq.QNetMultiModel) and len(model.getSubModels()) == 1:
                name = name.replace("val_", f"val_{model.getSubModels()[0].name}_")
            metrics[name.replace("val_", "valid_")] = value[0]
        else:
            if isinstance(model, mq.QNetMultiModel) and len(model.getSubModels()) == 1:
                name = f"{model.getSubModels()[0].name}_{name}"
            metrics[f"train_{name}"] = value[0]
    # delete redundant total loss if MultiModel is present
    if isinstance(model, mq.QNetMultiModel):
        if "train_loss" in metrics: 
            del metrics["train_loss"]
        if "valid_loss" in metrics: 
            del metrics["valid_loss"]          
    return metrics

# dummy function for the epoch callback
# function is called before epoch training starts
def epoch_callback_fn(model, epoch, history):
    pass
         
# fit a specified model until the last epoch is reached (last_epoch included)
def fit_model(model, last_epoch, datasets_fn, history = {}, epoch_callback_fn=None,
              save_folder_path=".", variables_file_name="model_variables.npz", optimizer_file_name="model_optimizer.npz",
              history_file_name="model_history.json", plots_folder_name="plots"):
    if len(history) > 0:
        start_epoch = history["epoch"][-1] + 1
    else:
        start_epoch = 0

    if not os.path.isdir(f"{save_folder_path}"): os.mkdir(f"{save_folder_path}")
    if not os.path.isdir(f"{save_folder_path}/{plots_folder_name}"): os.mkdir(f"{save_folder_path}/{plots_folder_name}")
    
    # temp seed that is generated from global seed before first epoch(global seed by set_seed function)         
    tmp_seed_1 = random.randint(0, 2**31 - 1)
    for epoch in range(start_epoch, last_epoch+1):        
        # generate a deterministic seed for each epoch
        tmp_seed_2 = random.Random(epoch).randint(0, 2**31 - 1)
        # set the epoch seed
        set_seed(tmp_seed_1 + tmp_seed_2)
        
        # if epoch not in history dict, add a new entry (entry is an empty list)
        if not "epoch" in history:
            history["epoch"] = []
        history["epoch"].append(epoch)
        
        # call the epoch callback function
        if epoch_callback_fn != None:
            epoch_callback_fn(model, epoch, history)
		
        # measure time for fitting and saving the model
        start_time = time.perf_counter()
        if epoch == 0:
            # evaluate untrained model
            epoch_metrics = evaluate_model(model, datasets_fn)
        else:
            # train model one epoch
            epoch_metrics = fit_model_epoch(model, datasets_fn)
        # save the weights of the model and optimizer temporary
        model.saveVariablesNPZ(f"{save_folder_path}/tmp_{variables_file_name}")
        model.saveOptimizerNPZ(f"{save_folder_path}/tmp_{optimizer_file_name}")
        # get the elapsed time
        elapsed_time = time.perf_counter() - start_time
        lr = model.optimizer.learning_rate.numpy()
        output_str = f"epoch: {epoch:>6}| time: {elapsed_time:8.2f}s| lr:{lr:1.8f}"
                        
        # add metrics to the output string and to the history dict
        for name, metric in epoch_metrics.items():
            output_str += f"| {name}: {metric:10.6f}"
            # if metric not in history dict, add a new entry (entry is an empty list)
            if not name in history:
                history[name] = []
            history[name].append(metric)
                
        # save the metrics temporay
        json.dump(history, open(f"{save_folder_path}/tmp_{history_file_name}", "w"))

        # get all metric types
        metric_type_names = []
        for entry_name in history.keys():
            if entry_name.startswith("train_"):
                metric_type_names.append(entry_name.rsplit("_", 1)[-1])
        
        # plot all metric types seperately             
        for metric_type_name in set(metric_type_names):
            plt.figure(figsize=(11.69, 8.27)) #A4 paper landscape mode
            for train_metric_name in history.keys():
                # get all train metric names
                if not train_metric_name.startswith("train_"):
                    continue
                # check if metric is of currently plotted type
                if not train_metric_name.endswith(f"_{metric_type_name}"):
                    continue

                train_label_short_name = train_metric_name.replace(f"_{metric_type_name}", "")
                plt.plot(history["epoch"], history[train_metric_name], label=train_label_short_name)

                color = matplotlib.colors.to_rgb(plt.gca().lines[-1].get_color())
                color = satuate_color(color, 0.45)
                plt.plot(history["epoch"], history[train_metric_name.replace("train_", "valid_")],
                         label=train_label_short_name.replace("train", "valid"), color=color)
                
            plt.gca().set_xlabel("epoch")
            plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            plt.gca().set_ylabel(metric_type_name)
            plt.gca().legend(bbox_to_anchor=(0.0, 1.06, 1.0, 0.102), loc="lower left",
                             ncol=4, mode="expand", borderaxespad=0.0)
            plt.tight_layout()
            plt.savefig(f"{save_folder_path}/{plots_folder_name}/{metric_type_name}.svg")
            plt.close()
                    
        # delete the old files and rename the temporary files
        if os.path.isfile(f"{save_folder_path}/{variables_file_name}"): os.remove(f"{save_folder_path}/{variables_file_name}")
        os.rename(f"{save_folder_path}/tmp_{variables_file_name}", f"{save_folder_path}/{variables_file_name}")

        if os.path.isfile(f"{save_folder_path}/{optimizer_file_name}"): os.remove(f"{save_folder_path}/{optimizer_file_name}")
        os.rename(f"{save_folder_path}/tmp_{optimizer_file_name}", f"{save_folder_path}/{optimizer_file_name}")
        
        if os.path.isfile(f"{save_folder_path}/{history_file_name}"): os.remove(f"{save_folder_path}/{history_file_name}")
        os.rename(f"{save_folder_path}/tmp_{history_file_name}", f"{save_folder_path}/{history_file_name}")
            
        # print to console and file
        print(output_str)
        # force write to file by flushing stdout
        sys.stdout.flush()
        # garbage collect to free some memory and prevent leaks
        gc.collect()
        
    return history


class EpochTimer(tf.keras.callbacks.Callback):
    """
    helper class to measure the elapsed time per epoch (training + validation times)
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs):
        self.epoch_times.append(time.perf_counter() - self.start_time)
        

def image_random_flip_lr(image, flip_probability=0.5):
    if tf.random.uniform([], 0.0, 1.0) < flip_probability:        
        return tf.image.flip_left_right(image)
    return image
      
def image_resize(image, size_h, size_w, resize_method):
    return tf.image.resize(image, [size_h, size_w], resize_method)

def image_resize_smaller_side(image, size, resize_method):
    image_h, image_w, _ = tf.unstack(tf.shape(image))
    aspect_ratio = tf.cast(image_w, tf.float32) / tf.cast(image_h, tf.float32)
    if image_h < image_w:
        size_h = size
        size_w = tf.cast(aspect_ratio * tf.cast(size_h, tf.float32), tf.int32)
    else:
        size_w = size
        size_h = tf.cast(tf.cast(size_w, tf.float32) / aspect_ratio, tf.int32)
    return image_resize(image, size_h, size_w, resize_method)

def image_center_crop(image, crop_h, crop_w):
    image_h, image_w, _ = tf.unstack(tf.shape(image))
    offset_h = (image_h - crop_h) // 2
    offset_w = (image_w - crop_w) // 2
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)

def image_random_crop(image, crop_h, crop_w):
    image_h, image_w, _ =  tf.unstack(tf.shape(image))
    offset_h = tf.random.uniform([], 0, image_h - crop_h + 1, tf.int32)
    offset_w = tf.random.uniform([], 0, image_w - crop_w + 1, tf.int32)
    return tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_h, crop_w)

def image_random_crop_scale_ratio(image, min_scale=0.08, max_scale=1.0, min_ratio=3/4, max_ratio=4/3):
    image_h, image_w, _ =  tf.unstack(tf.shape(image))
    for _ in tf.range(10): 
        scale = tf.random.uniform([], min_scale, max_scale, tf.float32)
        ratio = tf.math.exp(tf.random.uniform([], tf.math.log(min_ratio), tf.math.log(max_ratio), tf.float32))
        new_area = scale * tf.cast(image_h * image_w, tf.float32)    
        crop_h = tf.cast(tf.math.sqrt(new_area / ratio), tf.int32)
        crop_w = tf.cast(tf.math.sqrt(new_area * ratio), tf.int32)
        if 0 < crop_h and crop_h <= image_h and 0 < crop_w and crop_w <= image_w:
            image = image_random_crop(image, crop_h, crop_w)
            break
    return image

def image_random_brightness(image, min_delta, max_delta):
    delta = tf.random.uniform([], min_delta, max_delta, tf.float32)
    return tf.image.adjust_brightness(image, delta)

def image_random_hue(image, min_delta, max_delta):
    delta = tf.random.uniform([], min_delta, max_delta, tf.float32)
    return tf.image.adjust_hue(image, delta)

def image_random_contrast(image, min_factor, max_factor):
    contrast_factor = tf.random.uniform([], min_factor, max_factor, tf.float32) 
    return tf.image.adjust_contrast(image, contrast_factor)

def image_random_saturation(image, min_factor, max_factor):
    saturation_factor = tf.random.uniform([], min_factor, max_factor, tf.float32) 
    return tf.image.adjust_saturation(image, saturation_factor)

def to_value(valueOrTensor, dtype):
    if tf.is_tensor(valueOrTensor):
        value = valueOrTensor.numpy()
        assert isinstance(value.item(), dtype), "Incorrect type!" + str(value)
    else:
        value = valueOrTensor
    return value

def to_variable(valueOrVariable, dtype):
    if isinstance(valueOrVariable, tf.Variable):
        assert valueOrVariable.dtype == dtype, "Incorrect type!"
        return valueOrVariable
    return tf.Variable(valueOrVariable, synchronization=tf.VariableSynchronization.ON_WRITE, dtype=dtype, trainable=False)

def createLoggerEntry(type_base_name:str, type_name:str, name:str, params:dict, children:List[dict], loggers:List[dict]):
    type_ = {"base": type_base_name, "name": type_name}
    return {"name": name, "type": type_,
            "params": params,
            "children": children,
            "loggers": loggers}