# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.cm as cm


def make_gradcam_heatmap_via_grad_model(img_array, grad_model):
    """
    Generates a heatmap from the given grad_model

    This code is under the Apache License 2.0 and copied from this repo here:
    https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
    Args:
        img_array: some image with the correct dimensions [batch, h, w, c]
        grad_model: the gradmodel generated with the get_grad_model function

    Returns: a heatmap as a numpy array for each item in the batch (batched together for sure pal^^)

    """
    assert (len(np.shape(img_array)) == 4)  # has [batch, h, w, c] format
    #assert (np.shape(img_array)[3] == 3)  # has r, g, b channels

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        class_channel = preds[:]  # Was passiert wenn ich den ground truth einsetze?

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Generates a heatmap from the given model and name of the last conv layer
    WARNING: If you call this function repeatedly you get slow performance!
    Use the get_grad_model to get the grad_model once and call the make_gradcam_heatmap_via_grad_model
    repeatedly.
    Args:
        img_array: some image with the correct dimensions [batch, h, w, c]
        model: a model
        last_conv_layer_name: the name of the last conv layer in the model

    Returns: a heatmap as a numpy array for each item in the batch (batched together for sure pal^^)

    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = get_grad_model(model, last_conv_layer_name)
    return make_gradcam_heatmap_via_grad_model(img_array, grad_model)


def get_grad_model(model, last_conv_layer_name):
    """
    Generates the grad model so you can create nifty colorful images later
    with the make_gradcam_heatmap_via_grad_model
    Args:
        model: a model
        last_conv_layer_name: the last conv layer name in your model.
        What? You only got dense layers in your model? Bad luck ma dude it only works with conv layers!
        You lose positional information otherwise.

    Returns: the grad_model. Pass this to the make_gradcam_heatmap_via_grad_model

    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    return grad_model


def superimpose_heatmap_with_image(img, heatmap, alpha=0.9):
    """
    Superimposes some image with the given heatmap.
    You know, those nifty colorful images showing where it's hot in red and cold in blue?

    This code is under the Apache License 2.0 and copied from this repo here:
    https://github.com/keras-team/keras-io/blob/master/examples/vision/grad_cam.py
    Args:
        img: some image without the batch dimension and with rbg channels: [h, w, 3]
        heatmap: a single heatmap created with the make_gradcam_heatmap or make_gradcam_heatmap_via_grad_model function
        alpha: how much of the heatmap should be seen? Get closer to 1 to show more of the heatmap
        and less of the original image

    Returns: nifty colorful image showing where it's hot in red and cold in blue

    """
    if np.max(img) <= 1.0:
        img = img * 255

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return np.array(superimposed_img)