from tensorflow.keras import layers, models
from ml.tf2.capsnet import layers as capslayers
import numpy as np


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256,
                          kernel_size=9,
                          strides=1,
                          padding='valid',
                          activation='relu',
                          name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = capslayers.PrimaryCaps(dim_capsule=8,
                                         n_channels=32,
                                         kernel_size=9,
                                         strides=2,
                                         padding='valid',
                                         name='primarycaps')(conv1)

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = capslayers.CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                        name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = capslayers.Length(name='capsnet_tf2')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    # The true label is used to mask the output of capsule layer. For training
    masked_by_y = capslayers.Mask()([digitcaps, y])
    masked = capslayers.Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = capslayers.Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

