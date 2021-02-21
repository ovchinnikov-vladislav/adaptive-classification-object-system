from abc import ABC

import tensorflow as tf
import bmstu.capsnet.layers.basic as basic_layers
import bmstu.layers as common_layers
import bmstu.capsnet.layers.gamma as gamma_layers
import bmstu.capsnet.layers.matrix as matrix_layers
import bmstu.capsnet.metrics.gamma as gamma_metrics
from bmstu.capsnet import losses
from bmstu.utls import pgd


class CapsNet:
    def __init__(self, shape, classes, routings):
        self.shape = shape
        self.classes = classes

        self.input_capsnet = tf.keras.layers.Input(shape=shape)
        self.conv1 = tf.keras.layers.Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)
        self.primaryCaps = basic_layers.PrimaryCapsule2D(capsules=32, dim_capsules=8, kernel_size=9, strides=2)
        self.capsules = basic_layers.Capsule(capsules=classes, dim_capsules=16, routings=routings)
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


class GammaCapsNet(tf.keras.Model, ABC):
    # TODO: добить обучение Gamma СapsNet
    def __init__(self, shape, classes, routings, gamma_robust=True):
        super(GammaCapsNet, self).__init__()
        self.gamma_robust = gamma_robust
        self.classes = classes
        self.input_capsnet = tf.keras.layers.Input(shape=shape)
        self.conv1 = tf.keras.layers.Conv2D(256, (9, 9), padding='valid', activation=tf.nn.relu)
        self.primaryCaps = basic_layers.PrimaryCapsule2D(capsules=32, dim_capsules=8, kernel_size=9, strides=2)
        self.gammaCaps1 = gamma_layers.GammaCapsule(capsules=32, dim_capsules=8, routings=routings)
        self.gammaCaps2 = gamma_layers.GammaCapsule(capsules=10, dim_capsules=16, routings=routings)
        self.decoder = gamma_layers.GammaDecoder(dim=28)
        self.norm = common_layers.Norm()

        self.input_decoder = tf.keras.layers.Input(shape=(classes,))
        self.optimizer = self.train_accuracy = self.train_t_score = self.train_d_score = None
        self.test_accuracy = self.test_loss = self.test_t_score = self.test_d_score = None

    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        super(GammaCapsNet, self).compile(optimizer, loss, metrics, loss_weights,
                                          weighted_metrics, run_eagerly, **kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_t_score = tf.keras.metrics.Mean(name='train_t_score')
        self.train_d_score = tf.keras.metrics.Mean(name='train_d_score')

        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_t_score = tf.keras.metrics.Mean(name='test_t_score')
        self.test_d_score = tf.keras.metrics.Mean(name='test_d_score')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(self.input_capsnet)
        x = self.primaryCaps(x)
        v_1, c_1 = self.gammaCaps1(x)
        v_2, c_2 = self.gammaCaps2(v_1)

        r = self.decoder(v_2)
        out = self.norm(v_2)

        t_score = (gamma_metrics.t_score(c_1) + gamma_metrics.t_score(c_2)) / 2.0
        d_score = gamma_metrics.d_score(v_1)

        return out, r, [v_1, v_2], t_score, d_score

    def train_step(self, data):
        x, y = data
        x_adv = pgd(x, y, self, eps=0.1, a=0.01, k=40) if self.gamma_robust else x
        with tf.GradientTape() as tape:
            y_pred, reconstruction, _, t_score, d_score = self(x_adv, y)
            loss, _ = losses.compute_loss(y, y_pred, reconstruction, x)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.train_accuracy.update_state(y, y_pred)
        self.train_t_score.update_state(t_score)
        self.train_d_score.update_state(d_score)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred, reconstruction, layers, t_score, d_score = self(x, y)
        loss, _ = losses.compute_loss(y, y_pred, reconstruction, x)

        self.test_accuracy.update_state(y, y_pred)
        self.test_loss.update_state(loss)
        self.test_t_score.update_state(t_score)
        self.test_d_score.update_state(d_score)

        pred = tf.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(y, pred, num_classes=self.classes)

        return {m.name: m.result() for m in self.metrics}


class MatrixCapsNet:
    def __init__(self, shape, classes, routings, batch_size):
        self.shape = shape
        self.classes = classes
        self.routings = routings

        self.input_capsnet = tf.keras.layers.Input(shape=shape, batch_size=batch_size)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
        self.primaryCaps = matrix_layers.PrimaryCapsule2D(capsules=32, kernel_size=1, strides=1, padding='valid',
                                                          pose_shape=[4, 4])
        self.convCaps1 = matrix_layers.ConvolutionalCapsule(shape=[3, 3, 32, 32], strides=[1, 2, 2, 1],
                                                            routings=routings)

    def build(self):
        self.conv1 = self.conv1(self.input_capsnet)
        self.primaryCaps = self.primaryCaps(self.conv1)
        self.convCaps1 = self.convCaps1(self.primaryCaps)

        model = tf.keras.models.Model(self.input_capsnet, self.convCaps1)

        return model


if __name__ == '__main__':
    train_model = MatrixCapsNet(shape=[28, 28, 1], classes=10, routings=3, batch_size=24).build()
    train_model.summary()
