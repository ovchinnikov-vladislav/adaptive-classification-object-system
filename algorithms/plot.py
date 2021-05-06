import pandas
from matplotlib import pyplot as plt
import os


def plot_log(filename1, filename2, training_metric, val_metric, train_name1, train_name2, val_name1, val_name2,
             title, color1, color2, save_dir='.', show=True):
    data1 = pandas.read_csv(filename1)
    data2 = pandas.read_csv(filename2)

    fig = plt.figure()
    # plt.plot(0, 0)
    # plt.plot(0, 1)

    for key in data1.keys():
        if key == training_metric:
            plt.plot(data1['epoch'].values, data1[key].values, f'{color1}--', label=train_name1)
        if key == val_metric:
            plt.plot(data1['epoch'].values, data1[key].values, f'{color1}', label=val_name1)

    for key in data2.keys():
        if key == training_metric:
            plt.plot(data2['epoch'].values, data2[key].values, f'{color2}--', label=train_name2)
        if key == val_metric:
            plt.plot(data2['epoch'].values, data2[key].values, f'{color2}', label=val_name2)

    plt.legend()
    plt.title(title)

    fig.savefig(os.path.join(save_dir, f'log_{training_metric}_{val_metric}.png'))
    if show:
        plt.show()

    plt.close()
    return fig


if __name__ == '__main__':
    plot_log(filename1=os.path.join('other', 'capsnet-basic-v1-cifar10-logs', 'history_training_cifar10.csv'),
             filename2=os.path.join('other', 'complex-res-capsnet-3level-v3-cifar10-logs', 'history_training_cifar10.csv'),
             training_metric='length_accuracy',
             val_metric='val_length_accuracy',
             train_name1='Точность при обучении (исходная капсульная сеть)',
             train_name2='Точность при обучении (трехуровневая капсульная сеть)',
             val_name1='Точность при валидации (исходная капсульная сеть)',
             val_name2='Точность при валидации (трехуровневая капсульная сеть)',
             title='Точность при обучении и при валидации',
             color1='b', color2='g', show=True)

    plot_log(filename1=os.path.join('other', 'capsnet-basic-v1-cifar10-logs', 'history_training_cifar10.csv'),
             filename2=os.path.join('other', 'complex-res-capsnet-3level-v3-cifar10-logs', 'history_training_cifar10.csv'),
             training_metric='length_loss',
             val_metric='val_length_loss',
             train_name1='Потери при обучении (исходная капсульная сеть)',
             train_name2='Потери при обучении (трехуровневая капсульная сеть)',
             val_name1='Потери при валидации (исходная капсульная сеть)',
             val_name2='Потери при валидации (трехуровневая капсульная сеть)',
             title='Потери при обучении и при валидации',
             color1='b', color2='g', show=True)
