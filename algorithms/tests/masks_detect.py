from libs.capsnets.models.efficient import EfficientCapsuleNetwork
import tensorflow as tf
import numpy as np


if __name__ == '__main__':
    model = EfficientCapsuleNetwork(name=f'efficient_capsnet_masks')\
        .create(input_shape=(100, 100, 3), num_classes=2, dataset='masks', decoder=False)

    model.load_weights('./capsnet.h5')

    image = tf.keras.preprocessing.image.load_img('./No Mask116.jpg')
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    x = tf.image.resize(input_arr, (100, 100))

    y_pred = model.predict(x)

    if np.argmax(y_pred, 1) == 0:
        print('в маске')
    else:
        print('без маски')
