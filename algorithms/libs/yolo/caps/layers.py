import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, UpSampling2D, ZeroPadding2D, Concatenate, Layer, Reshape,
                                     Conv2D, BatchNormalization, LeakyReLU, Add, Lambda)
from tensorflow.keras.regularizers import l2


def _caps_conv2d(inputs, weights, filters_out, kernel_size, strides, padding, iterations):
    predictions = _predict(inputs, weights, filters_out, kernel_size, strides, padding)
    outputs = _route(predictions, iterations)

    return outputs


def _predict(inputs, weights, filters_out, kernel_size, strides, padding):
    batch_size = tf.shape(inputs)[0]
    _, height_in, width_in, filters_in, dims_in = inputs.shape

    inputs_flat = tf.reshape(inputs, shape=[-1, height_in, width_in, filters_in * dims_in], name='inputs_flat')

    inputs_patches = tf.image.extract_patches(
        inputs_flat,
        sizes=[1, kernel_size[0], kernel_size[1], 1],
        strides=[1, strides[0], strides[1], 1],
        rates=[1, 1, 1, 1],
        padding=padding.upper())

    _, height_out, width_out, _ = inputs_patches.shape

    caps_per_patch = kernel_size[0] * kernel_size[1] * filters_in

    inputs_patches = tf.reshape(inputs_patches, shape=[-1, height_out, width_out, 1, caps_per_patch, dims_in, 1], name='inputs_patches')

    inputs_patches_tiled = tf.tile(inputs_patches, multiples=[1, 1, 1, filters_out, 1, 1, 1], name='inputs_patches_tiled')

    W_tiled = tf.tile(weights, multiples=[batch_size, height_out, width_out, 1, 1, 1, 1], name='W_tiled')

    predictions = tf.matmul(W_tiled, inputs_patches_tiled, name='predictions')

    return predictions


def _route(predictions, iterations):
    batch_size = tf.shape(predictions)[0]
    _, height, width, filters_out, caps_per_patch, dims_out, _ = predictions.shape.as_list()

    initial_outputs = tf.zeros([batch_size, height, width, filters_out, 1, dims_out, 1], name='initial_outputs')

    logits = tf.zeros([batch_size, height, width, filters_out, caps_per_patch, 1, 1], name='logits')

    def _routing_iteration(loop_counter, logits_old, outputs_old):
        outputs_old_tiled = tf.tile(outputs_old, multiples=[1, 1, 1, 1, caps_per_patch, 1, 1], name='outputs_old_tiled')

        agreement = tf.matmul(predictions, outputs_old_tiled, transpose_a=True, name='agreement')

        logits = tf.add(logits_old, agreement, name='logits')

        coupling_coefficients = tf.nn.softmax(logits, axis=3, name='coupling_coefficients')

        weighted_predictions = tf.multiply(coupling_coefficients, predictions, name='weighted_predictions')

        centroids = tf.reduce_sum(weighted_predictions, axis=4, keepdims=True, name='centroids')

        outputs = squash(centroids, axis=-2)

        return loop_counter + 1, logits, outputs

    loop_counter = tf.constant(1, name='loop_counter')

    _, _, outputs_sparse = tf.while_loop(
        cond=lambda c, l, o: tf.less_equal(c, iterations),
        body=_routing_iteration,
        loop_vars=[loop_counter, logits, initial_outputs],
        name='routing_loop'
    )

    outputs = tf.squeeze(outputs_sparse, axis=[4, -1], name='outputs')

    return outputs


def norm(s, axis=-1, epsilon=1e-7, keepdims=False):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims, name='squared_norm')
    result = tf.sqrt(squared_norm + epsilon, name='norm')
    return result


def squash(s, axis=-1, epsilon=1e-7):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True, name='squared_norm')
    normal = tf.sqrt(squared_norm + epsilon, name='norm')
    squash_factor = tf.divide(squared_norm, (1 + squared_norm), name='squash_factor')
    unit_vector = tf.divide(s, normal, name='unit_vector')
    squashed = squash_factor * unit_vector
    return squashed


class PrimaryCapsule(Layer):
    def __init__(self, filters, dims, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dims = dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = None

    def build(self, input_shape):
        super().build(input_shape)
        self.conv = Conv2D(filters=self.filters * self.dims, kernel_size=self.kernel_size,
                           strides=self.strides, padding='valid', activation=tf.nn.relu)

        self.built = True

    def call(self, inputs, **kwargs):
        tf.assert_rank(inputs, rank=4, message='''`inputs` must be a tensor of feature maps (i.e. of shape
                                                  (batch_size, height, width, filters))''')

        x = self.conv(inputs)
        _, height, width, _ = x.shape

        capsules = tf.reshape(x, shape=[-1, height, width, self.filters, self.dims])

        return squash(capsules)

    def get_config(self):
        return super().get_config()


class DenseCapsule(Layer):
    def __init__(self, caps, dims, iterations=2, **kwargs):
        super().__init__(**kwargs)
        self.caps = caps
        self.dims = dims
        self.iterations = iterations
        self.filters_in = None
        self.dims_in = None
        self.W = None

    def build(self, input_shape):
        super().build(input_shape)

        inputs_rank = len(input_shape)

        if inputs_rank == 3:
            _, filters_in, dims_in = input_shape
        elif inputs_rank == 5:
            _, height, width, filters, dims_in = input_shape
            filters_in = height * width * filters
        else:
            raise Exception('''`inputs` must either be a flat tensor of capsules (i.e.
                                       of shape (batch_size, caps_in, dims_in)) or a tensor of
                                       capsule filters (i.e. of shape 
                                       (batch_size, height, width, filters, dims))''')

        self.filters_in = filters_in
        self.dims_in = dims_in

        caps_per_patch = 1 * 1 * filters_in

        self.W = self.add_weight(shape=[1, 1, 1, self.caps, caps_per_patch, self.dims, self.dims_in],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.1),
                                 name='W',
                                 trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):
        inputs_filters = Reshape(target_shape=[-1, 1, 1, self.filters_in, self.dims_in])(inputs)

        outputs = _caps_conv2d(inputs_filters, self.W, filters_out=self.caps, kernel_size=(1, 1),
                               strides=(1, 1), padding='valid', iterations=self.iterations)

        outputs = Reshape(target_shape=[-1, self.caps, self.dims])(outputs)

        return outputs

    def get_config(self):
        return super().get_config()


class ConvolutionalCapsule(Layer):
    def __init__(self, filters, dims, kernel_size, strides, padding='valid', iterations=2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dims = dims
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.iterations = iterations
        self.W = None

    def build(self, input_shape):
        super().build(input_shape)
        _, height, width, filters_in, dims_in = input_shape

        caps_per_patch = self.kernel_size * self.kernel_size * filters_in
        self.W = self.add_weight(shape=[1, 1, 1, self.filters, caps_per_patch, self.dims, dims_in],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.1),
                                 name='W',
                                 trainable=True)

        self.built = True

    def call(self, inputs, **kwargs):

        outputs = _caps_conv2d(inputs, self.W, self.filters, (self.kernel_size, self.kernel_size),
                               (self.strides, self.strides), self.padding, self.iterations)

        return outputs

    def get_config(self):
        return super().get_config()


def conv(x, filters, size, down_sampling=False,
         activation='leaky', batch_norm=True):
    def mish(inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))

    if down_sampling:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1

    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        if activation == 'mish':
            x = mish(x)
        elif activation == 'leaky':
            x = LeakyReLU(alpha=0.1)(x)

    return x


def residual(x, first_filters, second_filters, activation='leaky'):
    prev = x
    x = conv(x, first_filters, 1, activation=activation)
    x = conv(x, second_filters, 3, activation=activation)
    x = Add()([prev, x])
    return x


def block(x, filters, blocks):
    x = conv(x, filters, 3, down_sampling=True)
    for _ in range(blocks):
        x = residual(x, filters // 2, filters)
    return x


def conv_net(name=None, size=None, channels=3):
    x = inputs = Input([size, size, channels])
    x = conv(x, 32, 3)
    x = block(x, 64, 1)
    x = block(x, 128, 2)  # skip connection
    x = x_36 = block(x, 256, 8)  # skip connection
    x = x_61 = block(x, 512, 8)
    x = block(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def yolo_conv_input_tuple(x_in, filters):
    inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
    x, x_skip = inputs

    # concat with skip connection
    x = conv(x, filters, 1)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_skip])

    return x, inputs


def yolo_conv(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])

    x = conv(x, filters, 1)
    x = conv(x, filters * 2, 3)
    x = conv(x, filters, 1)
    x = conv(x, filters * 2, 3)
    x = conv(x, filters, 1)
    return Model(inputs, x, name=name)(x_in)


def yolo_conv_tiny(x_in, filters, name=None):
    if isinstance(x_in, tuple):
        x, inputs = yolo_conv_input_tuple(x_in, filters)
    else:
        x = inputs = Input(x_in.shape[1:])
        x = conv(x, filters, 1)

    return Model(inputs, x, name=name)(x_in)


def yolo_output(x_in, filters, grid, anchors, classes, name=None):
    x = inputs = Input(x_in.shape[1:])
    # x = conv(x, filters * 2, 3)
    # x = conv(x, anchors * (classes + 5), 1, batch_norm=False)

    x = PrimaryCapsule(16, 8, 3, 1)(x)
    capsules = ConvolutionalCapsule((grid - (x.shape[1] - 3 + 1)) * 2, anchors * (classes + 5), kernel_size=3, strides=1)(x)

    x = Lambda(lambda inp: tf.reshape(inp, (-1, grid, grid, anchors, classes + 5)))(capsules)
    model = tf.keras.Model(inputs, x, name=name)
    model.summary()
    return model(x_in)


def capsules_yolo(anchors, size, channels, classes, training=False):
    masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    x = inputs = Input([size, size, channels], name='input')
    x = conv(x, 32, 3)
    x = block(x, 64, 1)
    x = block(x, 128, 2)  # skip connection
    x = x_36 = block(x, 256, 8)  # skip connection
    x = x_61 = block(x, 512, 8)
    x = block(x, 1024, 4)

    x = yolo_conv(x, 512, name='yolo_conv_0')
    output_0 = yolo_output(x, 8, size // 32, len(masks[0]), classes, name='yolo_output_0')

    x = yolo_conv((x, x_61), 256, name='yolo_conv_1')
    output_1 = yolo_output(x, 8, size // 32 * 2, len(masks[1]), classes, name='yolo_output_1')

    x = yolo_conv((x, x_36), 128, name='yolo_conv_2')
    output_2 = yolo_output(x, 8, size // 32 * 4, len(masks[2]), classes, name='yolo_output_2')

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    from libs.yolo.utils import yolo_boxes, yolo_nms
    boxes_0 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda inp: yolo_boxes(inp, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda inp: yolo_nms(inp, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


if __name__ == '__main__':
    import config
    from libs.yolo.utils import get_anchors

    anchors = get_anchors(config.yolo_caps_anchors)

    model = capsules_yolo(anchors=anchors, size=416, channels=3, classes=1, training=True)
    model.summary()
