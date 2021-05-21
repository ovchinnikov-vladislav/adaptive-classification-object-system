from tensorflow.keras import layers
from libs.capsnets.layers.matrix import PrimaryCapsule3D, ConvolutionalCapsule3D, ClassCapsule
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    import config
    from skvideo.io import vread
    from scipy.misc import imresize

    class_names = [c.strip() for c in open(config.ucf24_classes_ru, 'r', encoding='utf8').readlines()]
    num_classes = len(class_names)

    shape = (8, 112, 112, 3)

    inputs = layers.Input(shape, name='input')
    conv1 = layers.Conv3D(64, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                          activation=tf.nn.relu, name='conv1')(inputs)
    conv2 = layers.Conv3D(128, kernel_size=[3, 3, 3], padding='same', strides=[1, 2, 2],
                          activation=tf.nn.relu, name='conv2')(conv1)
    conv3 = layers.Conv3D(256, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                          activation=tf.nn.relu, name='conv3')(conv2)
    conv4 = layers.Conv3D(256, kernel_size=[3, 3, 3], padding='same', strides=[1, 2, 2],
                          activation=tf.nn.relu, name='conv4')(conv3)
    conv5 = layers.Conv3D(512, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                          activation=tf.nn.relu, name='conv5')(conv4)
    conv6 = layers.Conv3D(512, kernel_size=[3, 3, 3], padding='same', strides=[1, 1, 1],
                          activation=tf.nn.relu, name='conv6')(conv5)
    prim_caps = PrimaryCapsule3D(channels=32, kernel_size=[3, 9, 9], strides=[1, 1, 1], padding='valid',
                                 name='prim_caps')(conv6)
    sec_caps = ConvolutionalCapsule3D(channels=32, kernel_size=[3, 5, 5], strides=[1, 2, 2], padding='valid',
                                      route_mean=True, name='sec_caps')(prim_caps)
    pred_caps = ClassCapsule(n_caps_j=num_classes, subset_routing=-1, route_min=0.0, name='pred_caps', coord_add=True,
                             ch_same_w=True)(sec_caps)
    digit_preds = tf.reshape(pred_caps[1], (-1, num_classes))

    model = tf.keras.Model(inputs, digit_preds)
    model.load_weights(config.ucf24_caps_model)

    video = vread('D:/tensorflow_datasets/UCF-101/UCF-101/HorseRiding/v_HorseRiding_g02_c05.avi')

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

    f_skip = 2
    predictions = []
    for i in range(0, n_frames, 8 * f_skip):
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
        pred = model.predict(x_batch[0])

        predictions.append(pred)

    predictions = np.concatenate(predictions, axis=0)
    predictions = predictions.reshape((-1, 24))
    fin_pred = np.mean(predictions, axis=0)
    print(class_names[int(np.argmax(fin_pred))])
