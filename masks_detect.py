from libs.capsnets.models.efficient import EfficientCapsuleNetwork
import tensorflow as tf
import numpy as np



if __name__ == '__main__':
    model = EfficientCapsuleNetwork(name=f'efficient_capsnet_masks')\
        .create(input_shape=(64, 64, 3), num_classes=2, dataset='masks', decoder=False)

    model.load_weights('./efficient_capsnet_train_test/efficient_capsnet_logs/efficient_capsnet_masks-result-2021-04-19-de3563db-1198-47d4-9f59-37ed809e11d0.h5')

    image = tf.keras.preprocessing.image.load_img('./188.jpg')
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    x = tf.image.resize(input_arr, (64, 64))

    y_pred = model.predict(x)

    if np.argmax(y_pred, 1) == 0:
        print('в маске')
    else:
        print('без маски')
