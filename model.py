import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

import numpy as np
import pandas as pd

import os

from config import IMAGE_SHAPE, NUM_KEYPOINTS


def get_custom_model():
    resnet_v2_backbone = ResNet50V2(weights='imagenet',
                                    input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3),
                                    include_top=False,
                                    classifier_activation=None)     # Classifier activation needs to be removed for regression task
    resnet_v2_backbone.trainable = False

    inputs = layers.Input(shape=IMAGE_SHAPE)

    # This Conv2D layer was added to make the image 3 channel
    # Comparison with filling 3 channels with the same data?
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
    x = layers.Activation('relu')(x)

    x = preprocess_input(x)
    x = resnet_v2_backbone(x)
    x = layers.Dropout(0.1)(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    outputs = layers.Dense(30)(x)

    return Model(inputs, outputs, name='Model_V1')
