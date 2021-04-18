import tensorflow as tf
from libs.capsnets.layers.basic import Length, Decoder
from libs.capsnets.layers.efficient import PrimaryCapsule2D, Capsule
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras import Model
from libs.utls import BaseModelForTraining


class EfficientCapsuleNetwork(BaseModelForTraining):
    def create(self, input_shape, **kwargs):
        dataset = kwargs.get('dataset')
        decoder = kwargs.get('decoder')
        num_classes = kwargs.get('num_classes')

        if decoder:
            self.is_decoder = True

        x = inputs = Input(shape=input_shape)

        x = tf.keras.layers.Conv2D(32, 5, activation="relu", padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, 3, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if dataset == 'cifar10':
            x = PrimaryCapsule2D(128, 9, 72, 16)(x)
        elif dataset == 'masks':
            x = PrimaryCapsule2D(128, 9, 1922, 256)(x)
        else:
            x = PrimaryCapsule2D(128, 9, 16, 8)(x)

        capsules = Capsule(num_classes, 16)(x)
        output = Length(name='length')(capsules)

        if self.is_decoder:
            input_decoder = Input(shape=(num_classes,))

            decoder = Decoder(num_classes=num_classes, output_shape=input_shape, name='decoder')

            train_model = Model([inputs, input_decoder], [output, decoder([capsules, input_decoder])], name=self.name)

            eval_model = Model(inputs, [output, decoder(capsules)], name=self.name)

            return train_model, eval_model
        else:
            train_model = Model(inputs, output, name=self.name)
            return train_model
