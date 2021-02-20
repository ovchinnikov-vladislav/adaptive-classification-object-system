import tensorflow as tf
import bmstu.capsnet.layers.basic as basic_layers


class CapsNet:
    def __init__(self, shape, classes, name=''):
        super(CapsNet, self).__init__(name)
        self.shape = shape
        self.classes = classes

        self.input_capsnet = tf.keras.layers.Input(shape=shape)
        self.conv1 = tf.keras.layers.Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)
        self.primaryCaps = basic_layers.PrimaryCapsule(capsules=32, dim_capsules=8, kernel_size=9, strides=2)
        self.capsules = basic_layers.Capsule(capsules=classes, dim_capsules=16, routings=3)
        self.output = basic_layers.Length()

        self.input_decoder = tf.keras.layers.Input(shape=(classes,))
        self.input_noise_decoder = tf.keras.layers.Input(shape=(classes, 16))

    def build(self):
        self.conv1 = self.conv1(self.input_capsnet)
        self.primaryCaps = self.primaryCaps(self.conv1)
        self.capsules = self.capsules(self.primaryCaps)
        self.output = self.output(self.capsules)

        train_model = tf.keras.models.Model(
            [self.input_capsnet, self.input_decoder],
            [self.output, basic_layers.Decoder(
                classes=self.classes, output_shape=self.shape)([self.capsules, self.input_decoder])])

        eval_model = tf.keras.models.Model(
            self.input_capsnet,
            [self.output, basic_layers.Decoder(classes=self.classes, output_shape=self.shape)(self.capsules)])

        noised_digitcaps = tf.keras.layers.Add()([self.capsules, self.input_noise_decoder])
        manipulate_model = tf.keras.models.Model(
            [self.input_capsnet, self.input_decoder, self.input_noise_decoder],
            basic_layers.Decoder(classes=self.classes, output_shape=self.shape)([noised_digitcaps, self.input_decoder]))

        return train_model, eval_model, manipulate_model
