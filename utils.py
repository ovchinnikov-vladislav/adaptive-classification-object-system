from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.utils import to_categorical
import cv2
import os
import numpy as np


def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


# def video_capturing_function(datasets, folder_name):
#     for i in np.arange(len(datasets)):
#         video_name = datasets.video_name[i]
#         video_read_path = os.path.join('data', video_name)
#         cap = cv2.VideoCapture(video_read_path)
#         try:
#             os.mkdir(os.path.join(os.path.join(stretches_path, folder_name),
#                                   video_name.split(".")[0]))
#         except:
#             print("File Already Created")
#
#         train_write_file = os.path.join(os.path.join(stretches_path, folder_name),
#                                         video_name.split(".")[0])
#         cap.set(cv2.CAP_PROP_FPS, 1)
#         frameRate = cap.get(5)
#         x = 1
#         count = 0
#         while (cap.isOpened()):
#             frameId = cap.get(1)  # current frame number
#             ret, frame = cap.read()
#             if (ret != True):
#                 break
#             if (frameId % math.floor(frameRate) == 0):
#                 filename = "frame%d.jpg" % count
#                 count += 1
#                 frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite(os.path.join(train_write_file, filename), frame_grey)
#         cap.release()
#     return print("All frames written in the: " + folder_name + " Folder")
