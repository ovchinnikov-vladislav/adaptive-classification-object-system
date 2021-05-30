import tensorflow as tf
from tensorflow.keras.backend import epsilon, ndim, expand_dims
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Input, Conv3D
from libs.capsnets.layers.matrix import PrimaryCapsule3D, ConvolutionalCapsule3D, ClassCapsule
import config
import numpy as np
from scipy.misc import imresize


# v = ((||sj||^2) / (1 + ||sj||^2)) * (sj / ||sj||)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)

    additional_squashing = s_squared_norm / (1 + s_squared_norm)
    unit_scaling = vectors / tf.sqrt(s_squared_norm + epsilon())

    return additional_squashing * unit_scaling


# v = (1 - 1 / e^(||s||)) * (s / ||s||)
def efficient_squash(vectors, axis=-1, eps=10e-21):
    s_norm = tf.norm(vectors, axis=axis, keepdims=True)
    return (1 - 1 / (tf.math.exp(s_norm) + eps)) * (vectors / (s_norm + eps))


def own_batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if axes is None:
        # behaves like tf.batch_matmul as default
        axes = [x_ndim - 1, y_ndim - 2]
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = array_ops.reshape(y,
                              array_ops.concat(
                                  [array_ops.shape(y), [1] * diff], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = array_ops.reshape(x,
                              array_ops.concat(
                                  [array_ops.shape(x), [1] * diff], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
        else:
            out = math_ops.reduce_sum(
                math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
    else:
        adj_x = None if axes[0] == ndim(x) - 1 else True
        adj_y = True if axes[1] == ndim(y) - 1 else None
        out = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = array_ops.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = expand_dims(out, 1)
    return out


class VideoClassCapsNetModel:
    def __init__(self):
        self.class_names = [c.strip() for c in open(config.ucf24_classes_ru, 'r', encoding='utf8').readlines()]
        num_classes = len(self.class_names)

        shape = (8, 112, 112, 3)

        inputs = Input(shape, name='input')
        conv1 = Conv3D(filters=64, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                       activation=tf.nn.relu, name='conv1')(inputs)
        conv2 = Conv3D(filters=128, kernel_size=[3, 3, 3], padding='same', strides=[1, 2, 2],
                       activation=tf.nn.relu, name='conv2')(conv1)
        conv3 = Conv3D(filters=256, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                       activation=tf.nn.relu, name='conv3')(conv2)
        conv4 = Conv3D(filters=256, kernel_size=[3, 3, 3], padding='same', strides=[1, 2, 2],
                       activation=tf.nn.relu, name='conv4')(conv3)
        conv5 = Conv3D(filters=512, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                       activation=tf.nn.relu, name='conv5')(conv4)
        conv6 = Conv3D(filters=512, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                       activation=tf.nn.relu, name='conv6')(conv5)
        prim_caps = PrimaryCapsule3D(channels=32, kernel_size=[3, 9, 9], strides=[1, 1, 1], padding='valid',
                                     name='prim_caps')(conv6)
        sec_caps = ConvolutionalCapsule3D(channels=32, kernel_size=[3, 5, 5], strides=[1, 2, 2], padding='valid',
                                          route_mean=True, name='sec_caps')(prim_caps)
        pred_caps = ClassCapsule(n_caps_j=num_classes, subset_routing=-1, route_min=0.0, name='pred_caps',
                                 coord_add=True,
                                 ch_same_w=True)(sec_caps)
        digit_preds = tf.reshape(pred_caps[1], (-1, num_classes))

        self.model = tf.keras.Model(inputs, digit_preds)
        self.model.load_weights(config.ucf24_caps_model)

    def predict(self, video):
        n_frames = video.shape[0]
        crop_size = (112, 112)

        # assumes a given aspect ratio of (240, 320). If given a cropped video, then no resizing occurs
        if video.shape[1] != 112 and video.shape[2] != 112:
            h, w = 120, 160

            video_res = np.zeros((n_frames, 120, 160, 3))

            for f in range(n_frames):
                video_res[f] = imresize(video[f], (120, 160))
        else:
            h, w = 112, 112
            video_res = video

        # crops video to 112x112
        margin_h = h - crop_size[0]
        h_crop_start = int(margin_h / 2)
        margin_w = w - crop_size[1]
        w_crop_start = int(margin_w / 2)
        video_cropped = video_res[:, h_crop_start:h_crop_start + crop_size[0], w_crop_start:w_crop_start + crop_size[1], :]

        video_cropped = video_cropped / 255.

        f_skip = 1
        predictions = []
        for i in range(0, n_frames, f_skip):
            # if frames are skipped (subsampled) during training, they should also be skipped at test time
            # creates a batch of video clips
            x_batch = [[] for p in range(f_skip)]
            for k in range(f_skip * 8):
                if i + k >= n_frames:
                    x_batch[k % f_skip].append(np.zeros_like(video_cropped[-1]))
                else:
                    x_batch[k % f_skip].append(video_cropped[i + k])
            x_batch = [np.stack(x, axis=0) for x in x_batch]
            x_batch = [np.expand_dims(x, axis=0) for x in x_batch]

            # runs the network to get segmentations
            pred = self.model.predict(x_batch[0])

            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)
        predictions = predictions.reshape((-1, 24))
        fin_pred = np.mean(predictions, axis=0)
        print(fin_pred)

        return self.class_names[int(np.argmax(fin_pred))]
