from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import matplotlib.pyplot as plt
import pandas
import math
import numpy as np


def load(dataset):
    (x_train, y_train), (x_test, y_test) = (None, None), (None, None)
    shape = None
    if dataset == 'mnist':
        shape = [-1, 28, 28, 1]
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
        shape = [-1, 28, 28, 1]
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset == 'cifar10':
        shape = [-1, 32, 32, 3]
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar100':
        shape = [-1, 32, 32, 3]
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        raise Exception('undefined name dataset')

    x_train = x_train.reshape(shape).astype('float32') / 255.
    x_test = x_test.reshape(shape).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num) / width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num) / height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


def plot_log(filename, show=True):
    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    fig.savefig('result/log.png')
    if show:
        plt.show()
