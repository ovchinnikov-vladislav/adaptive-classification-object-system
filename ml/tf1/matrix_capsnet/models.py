from keras import layers, models, regularizers
from ml.tf1.matrix_capsnet import layers as matrix_caps_layers


def MatrixCapsNet(input_shape, num_classes, routings, batch_size):
    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=64,
                          kernel_size=5,
                          strides=2,
                          padding='same',
                          activation='relu',
                          name='conv1')(x)

    spatial_size = int(conv1.shape[1])
    assert conv1.shape[1:] == [spatial_size, spatial_size, 64]

    primary_caps = matrix_caps_layers.PrimaryCaps(num_capsules=8,
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding='valid')(conv1)

    l2 = regularizers.l2(0.0000002)

    conv_caps1 = matrix_caps_layers.ConvCapsules(kernel_size=3,
                                                 batch_size=batch_size,
                                                 strides=2,
                                                 num_capsules=16,
                                                 name='conv_caps1',
                                                 routings=routings,
                                                 weights_regularizer=l2)(primary_caps)

    conv_caps2 = matrix_caps_layers.ConvCapsules(kernel_size=3,
                                                 strides=1,
                                                 num_capsules=16,
                                                 batch_size=batch_size,
                                                 routings=routings,
                                                 weights_regularizer=l2)(conv_caps1)

    class_caps = matrix_caps_layers.ClassCapsules(conv_caps2, num_capsules=num_classes,
                                                  batch_size=batch_size,
                                                  routings=routings,
                                                  weights_regularizer=l2)

    model = models.Model(x, class_caps)
    model.summary()
    return model


if __name__ == '__main__':
    model = MatrixCapsNet(input_shape=[28, 28, 1], num_classes=5, routings=3, batch_size=64)

    from keras.datasets import mnist
    from keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    # print(x_train[0].reshape(1, 28, 28, 1))
