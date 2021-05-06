from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Conv2D, Input
import tensorflow as tf
from libs.capsnets.losses import spread_loss
from libs.capsnets.metrics.matrix import matrix_accuracy
from libs.capsnets.layers.matrix import PrimaryCapsule2D, ConvolutionalCapsule, ClassCapsule
from libs import utls


class MatrixCapsuleModel(Model):
    def call(self, inputs, training=None, mask=None):
        super(MatrixCapsuleModel, self).__init__(inputs, training, mask)

    def get_config(self):
        return super(MatrixCapsuleModel, self).get_config()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            activation, pose = self(x, training=True)
            loss = spread_loss(activation, pose, x, y, self.optimizer.learning_rate(self.optimizer.iterations))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, activation)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['spread_loss'] = loss
        return metrics

    def test_step(self, data):
        x, y = data
        activation, pose = self(x, training=False)
        loss = spread_loss(activation, pose, x, y, self.optimizer.learning_rate(self.optimizer.iterations))

        self.compiled_metrics.update_state(y, activation)

        metrics = {m.name: m.result() for m in self.metrics}
        metrics['spread_loss'] = loss
        return metrics


def matrix_em_capsnet(shape, classes, routings, batch_size, coord_add):
    inputs = Input(shape=shape, batch_size=batch_size)
    x = Conv2D(filters=32, kernel_size=[5, 5], strides=2,
               padding='same', activation=relu)(inputs)
    x = PrimaryCapsule2D(capsules=8, kernel_size=[1, 1], strides=1,
                         padding='valid', pose_shape=[4, 4])(x)
    x = ConvolutionalCapsule(capsules=16, kernel=3, stride=2, routings=routings)(x)
    x = ConvolutionalCapsule(capsules=16, kernel=3, stride=1, routings=routings)(x)
    output = ClassCapsule(capsules=classes, routings=routings, coord_add=coord_add)(x)

    return MatrixCapsuleModel(inputs, output)


if __name__ == '__main__':
    import numpy as np

    coord_add = [[[8., 8.], [12., 8.], [16., 8.]],
                 [[8., 12.], [12., 12.], [16., 12.]],
                 [[8., 16.], [12., 16.], [16., 16.]]]

    coord_add = np.array(coord_add, dtype=np.float32) / 28.

    (x_train, y_train), (x_test, y_test) = utls.load('fashion_mnist')
    x_val = x_test[:9000]
    y_val = y_test[:9000]
    x_test = x_test[9000:]
    y_test = y_test[9000:]

    epochs = 2
    batch_size = 25

    model = matrix_em_capsnet([28, 28, 1], 10, 3, batch_size, coord_add)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[(len(x_train) // batch_size * x) for x in
                    range(1, 8)],
        values=[x / 10.0 for x in range(2, 10)])),
        metrics=matrix_accuracy)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_val, y_val))

    print(y_test[0])
    activation, pose_out = model.predict(x_test[0])
    print(activation)
