import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import matplotlib.pyplot as plt
import pandas
import math
import numpy as np
import io
import itertools


def pgd(x, y, model, eps=0.3, k=40, a=0.01):
    """ Projected gradient descent (PGD) attack
    """
    x_adv = tf.identity(x)
    loss_fn = tf.nn.softmax_cross_entropy_with_logits

    for _ in range(k):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_adv)
            y_pred, _, _, _, _ = model(x_adv, y)
            classes = tf.shape(y_pred)[1]
            labels = tf.one_hot(y, classes)
            loss = loss_fn(labels=labels, logits=y_pred)
        dl_dx = tape.gradient(loss, x_adv)
        x_adv += a * tf.sign(dl_dx)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

    print('Finished attack', flush=True)
    return x_adv


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


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('blue'))
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicated label')
    return figure


def plot_generated_image(x, y_pred):
    classes = tf.shape(y_pred)
    y_pred = y_pred.numpy()

    figure = plt.figure()
    plt.imshow(x, cmap=plt.get_cmap('gray'))
    text = ""
    for i in range(classes):
        text += "%d: %.2f " % (i, y_pred[i])
        if (i + 1) % 5 == 0 and i > 0:
            text += "\n"

    plt.title(text)
    return figure


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
