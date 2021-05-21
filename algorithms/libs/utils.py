import io
import itertools
import math
import os
import datetime
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import uuid
from tensorflow.keras import callbacks
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical, plot_model


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


def plot_log(filename, training_metric, val_metric, train_name, val_name, title, color, save_dir='.', show=True):
    data = pandas.read_csv(filename)

    fig = plt.figure()

    for key in data.keys():
        if key == training_metric:
            plt.plot(data['epoch'].values, data[key].values, f'{color}--', label=train_name)
        if key == val_metric:
            plt.plot(data['epoch'].values, data[key].values, f'{color}', label=val_name)
    plt.legend()
    plt.title(title)

    fig.savefig(os.path.join(save_dir, f'log_{training_metric}_{val_metric}.png'))
    if show:
        plt.show()

    plt.close()
    return fig


def plot_history(history, training_metric, val_metric, train_name, val_name, title, show=True):
    train = history[training_metric]
    val = history[val_metric]
    epochs = range(1, len(train) + 1)

    figure = plt.figure()
    plt.plot(epochs, train, 'y--', label=train_name)
    plt.plot(epochs, val, 'y', label=val_name)
    plt.title(title)
    plt.legend()
    plt.close()
    if show:
        figure.show()

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def plot_dataset_images(x_batch, y_batch, n_img, class_names, show=True):
    max_c = 5  # max images per row

    if n_img <= max_c:
        r = 1
        c = n_img
    else:
        r = int(np.ceil(n_img / max_c))
        c = max_c

    fig, axes = plt.subplots(r, c, figsize=(15, 15))
    axes = axes.flatten()
    for img_batch, label_batch, ax in zip(x_batch, y_batch, axes):
        ax.imshow(img_batch, cmap='gray')
        ax.grid()
        ax.set_title('Class: {}'.format(class_names[np.argmax(label_batch)]))
    plt.tight_layout()
    if show:
        plt.show()
    plt.close()

    return fig


def plot_confusion_matrix(cm, class_names, title='Матрица ошибок', save_dir='.', show=True):
    figure = plt.figure(figsize=(8, 8))
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Greys'), vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = cm.max(initial=0) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('Входные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.close()
    if show:
        figure.show()
    figure.savefig(os.path.join(save_dir, f'confusion_matrix.png'))
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
    plt.close()
    return figure


def load(dataset):
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


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names):
    """Prints tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
      file_name: Name of the checkpoint file.
      tensor_name: Name of the tensor in the checkpoint file to print.
      all_tensors: Boolean indicating whether to print all tensors.
      all_tensor_names: Boolean indicating whether to print all tensor names.
    """
    pywrap_tensorflow = tf.compat.v1.train
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors or all_tensor_names:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key)
                if all_tensors:
                    print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
            It's likely that this is a V2 checkpoint and you need to provide the filename
            *prefix*.  Try removing the '.' and extension.  Try:
            inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


class ClassificationReportPlotWriter:
    @staticmethod
    def __show_values(pc, fmt="%.2f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

    @staticmethod
    def __cm2inch(*t):
        inch = 2.54
        if type(t[0]) == tuple:
            return tuple(i / inch for i in t[0])
        else:
            return tuple(i / inch for i in t)

    @staticmethod
    def __heatmap(auc, title, x_label, y_label, x_tick_labels, y_tick_labels, show,
                  figure_width=40, figure_height=20, correct_orientation=False, cmap='Greys', save_dir='.'):
        fig, ax = plt.subplots()
        c = ax.pcolor(auc, edgecolors='k', linestyle='dashed', linewidths=0.2, cmap=cmap)  # , vmin=0.3, vmax=0.9)

        ax.set_yticks(np.arange(auc.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(auc.shape[1]) + 0.5, minor=False)

        ax.set_xticklabels(x_tick_labels, minor=False)
        ax.set_yticklabels(y_tick_labels, minor=False)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.xlim((0, auc.shape[1]))

        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
        for t in ax.yaxis.get_major_ticks():
            t.tick1line.set_visible(True)
            t.tick2line.set_visible(True)
        plt.colorbar(c)
        ClassificationReportPlotWriter.__show_values(c)
        if correct_orientation:
            ax.invert_yaxis()
            ax.xaxis.tick_top()

        fig = plt.gcf()
        fig.set_size_inches(ClassificationReportPlotWriter.__cm2inch(figure_width, figure_height))
        plt.close()
        if show:
            fig.show()

        fig.savefig(os.path.join(save_dir, f'classification_report.png'))

        return fig

    @staticmethod
    def plot(classification_report, title='Отчет классификации', cmap='gist_gray', show=True):
        lines = classification_report.split('\n')

        classes = []
        plot_mat = []
        support = []
        class_names = []
        for line in lines[2: (len(lines) - 2)]:
            t = line.strip().split()
            if len(t) < 2:
                break
            classes.append(t[0])
            v = [float(x) for x in t[1: len(t) - 1]]
            support.append(int(t[-1]))
            class_names.append(t[0])
            plot_mat.append(v)

        x_label = 'Метрики'
        y_label = 'Классы'
        x_tick_labels = ['Точность (precision)', 'Полнота (recall)', 'F1-score']
        y_tick_labels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup in enumerate(support)]
        figure_width = 25
        figure_height = len(class_names) + 7
        correct_orientation = False
        return ClassificationReportPlotWriter.__heatmap(np.array(plot_mat), title, x_label, y_label, x_tick_labels,
                                                        y_tick_labels, show, figure_width, figure_height,
                                                        correct_orientation, cmap=cmap)


class BaseModelForTraining(ABC):
    def __init__(self, name):
        self.models = None
        self.training_model = None
        self.input_shape = None
        self.is_decoder = None
        self.name = name

    @abstractmethod
    def create(self, input_shape, **kwargs):
        pass

    def build(self, input_shape, **kwargs):
        self.input_shape = input_shape

        self.models = self.create(input_shape, **kwargs)

        if type(self.models) is tuple:
            self.training_model = self.models[0]
        else:
            self.training_model = self.models

        self.training_model.summary(line_length=200)

        return self.models

    def compile(self, **kwargs):
        self.training_model.compile(**kwargs)

    def __train_generator(self, x, y, batch_size, shift_fraction=0.):
        train_data_generator = ImageDataGenerator(width_shift_range=shift_fraction,
                                                  height_shift_range=shift_fraction)
        generator = train_data_generator.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            if self.is_decoder:
                yield [x_batch, y_batch], [y_batch, x_batch]
            else:
                yield x_batch, y_batch

    def __test_generator(self, x, y):
        if self.is_decoder:
            return [x, y], [y, x]
        else:
            return x, y

    def fit(self, x, y, batch_size, epochs, call_backs=None, load_weights=None, validation_data=None,
            set_plot_model=True, set_tensor_board=True, set_debug=False, set_model_checkpoint=True,
            set_csv_logger=True, log_dir='./', save_weights=True, checkpoint_monitor='accuracy'):
        if call_backs is None:
            call_backs = []
        cb = []
        cb += call_backs

        if not os.path.exists(os.path.join(log_dir, 'models')):
            os.makedirs(os.path.join(log_dir, 'models'))

        if set_csv_logger:
            cb.append(callbacks.CSVLogger(os.path.join(log_dir, 'history_training.csv')))

        if set_tensor_board:
            cb.append(callbacks.TensorBoard(log_dir=os.path.join(log_dir, 'tb'),
                                            batch_size=batch_size, histogram_freq=set_debug))
        if set_model_checkpoint:
            date = str(datetime.datetime.now()).split(' ')[0]
            file_name = f'{self.name}-{date}' + '-{epoch:02d}.h5'
            cb.append(callbacks.ModelCheckpoint(
                os.path.join(log_dir, 'models', file_name), monitor=checkpoint_monitor,
                save_best_only=True, save_weights_only=True, verbose=1))

        if set_plot_model:
            plot_model(self.training_model, to_file=os.path.join(log_dir, self.name + '.png'), show_shapes=True)

        if load_weights:
            self.training_model.load_weights(load_weights)

        history = self.training_model.fit(self.__train_generator(x, y, batch_size, 0.1), epochs=epochs,
                                          validation_data=self.__test_generator(validation_data[0], validation_data[1]),
                                          steps_per_epoch=int(y.shape[0] / batch_size), callbacks=cb)

        if save_weights:
            date = str(datetime.datetime.now()).split(' ')[0]
            self.training_model.save_weights(os.path.join(log_dir, f'{self.name}-result-{date}-{str(uuid.uuid4())}.h5'))

        return history

    def fit_generator(self, train_data, steps_per_epoch, epochs, validation_data=None, call_backs=None,
                      load_weights=None, set_plot_model=True, set_tensor_board=True,
                      set_debug=False, set_model_checkpoint=True, set_csv_logger=True, log_dir='./',
                      save_weights=True, checkpoint_monitor='accuracy'):
        if call_backs is None:
            call_backs = []
        cb = []
        cb += call_backs

        if not os.path.exists(os.path.join(log_dir, 'models')):
            os.makedirs(os.path.join(log_dir, 'models'))

        if set_csv_logger:
            cb.append(callbacks.CSVLogger(os.path.join(log_dir, 'history_training.csv')))

        if set_tensor_board:
            cb.append(callbacks.TensorBoard(log_dir=os.path.join(log_dir, 'tb'),
                                            batch_size=steps_per_epoch // epochs, histogram_freq=set_debug))
        if set_model_checkpoint:
            date = str(datetime.datetime.now()).split(' ')[0]
            file_name = f'{self.name}-{date}' + '-{epoch:02d}.h5'
            cb.append(callbacks.ModelCheckpoint(
                os.path.join(log_dir, 'models', file_name), monitor=checkpoint_monitor,
                save_best_only=True, save_weights_only=True, verbose=1))

        if set_plot_model:
            plot_model(self.training_model, to_file=os.path.join(log_dir, self.name + '.png'), show_shapes=True)

        if load_weights:
            self.training_model.load_weights(load_weights)

        history = self.training_model.fit(train_data, validation_data=validation_data,
                                          epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=cb)

        if save_weights:
            date = str(datetime.datetime.now()).split(' ')[0]
            self.training_model.save_weights(os.path.join(log_dir, f'{self.name}-result-{date}-{str(uuid.uuid4())}.h5'))

        return history
