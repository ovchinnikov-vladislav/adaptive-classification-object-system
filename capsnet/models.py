from tensorflow.keras import layers, models
import capsnet.layers as capslayers
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
    primarycaps = capslayers.PrimaryCaps(conv1,
                                         dim_capsule=8,
                                         n_channels=32,
                                         kernel_size=9,
                                         strides=2,
                                         padding='valid',
                                         name='primarycaps')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = capslayers.CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                                        name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = capslayers.Length(name='capsnet')(digitcaps)

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


def MatrixCapsNet(input_shape, batch_size, n_class, routings):
    x = layers.Input(shape=input_shape)
    print(x.shape)

    conv1 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu', name='conv1')(x)
    print(conv1.shape)

    primarycaps = capslayers.PrimaryCaps(conv1,
                                         dim_capsule=32,
                                         kernel_size=1,
                                         strides=1,
                                         padding='valid',
                                         pose_shape=[4, 4],
                                         name='primarycaps')

    convcaps1 = capslayers.ConvCapsuleLayer(primarycaps,
                                            shape=[3, 3, 32, 32],
                                            strides=[1, 2, 2, 1],
                                            routings=routings,
                                            batch_size=batch_size)

    convcaps2 = capslayers.ConvCapsuleLayer(convcaps1,
                                            shape=[3, 3, 32, 32],
                                            strides=[1, 1, 1, 1],
                                            routings=routings,
                                            batch_size=batch_size)

    classcaps = capslayers.ClassCapsuleLayer(num_classes=n_class,
                                             routings=routings,
                                             batch_size=batch_size)(convcaps2)

    train_model = models.Model(x, [poses, activations])

    return train_model, None, None


if __name__ == '__main__':
    from keras.datasets import mnist
    from keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    model, eval_model, manipulate_model = MatrixCapsNet(input_shape=x_train.shape[1:],
                                                        n_class=len(np.unique(np.argmax(y_train, 1))),
                                                        routings=3,
                                                        batch_size=100)

    from keras import optimizers

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics='accuracy')

    model.summary()
    model.fit([x_train, y_train], [y_train, x_train], epochs=20, batch_size=100,
              validation_data=[[x_test, y_test], [y_test, x_test]])
