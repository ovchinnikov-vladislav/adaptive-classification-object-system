from libs import utils
from libs.capsnets.models.complex import (CapsuleNetworkWith3Level, CapsuleNetworkWith4Level,
                                          ResCapsuleNetworkWith3LevelV1, ResCapsuleNetworkWith3LevelV2,
                                          Resnet50WithCapsuleNetworkWith3Level, ResCapsuleNetworkWith3LevelV3)
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--routings', default=3)
parser.add_argument('--save_dir', default='capsnet_logs')
parser.add_argument('--path_model', default='res_capsnet_3level_v3_cifar10-result-2021-04-25-59d24cba-68da-4ccd-976e-f246d1eba47d.h5')
parser.add_argument('--dataset', default='cifar10', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--model', default='res_capsnet_3level_v3', help='value: capsnet_3level, capsnet_4level, '
                                       'res_capsnet_3level_v1, res_capsnet_3level_v2, res50_capsnet_3level')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utils.load(args.dataset)

    if args.model == 'capsnet_4level':
        _, model = CapsuleNetworkWith4Level(name=f'capsnet_4level_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    elif args.model == 'res_capsnet_3level_v1':
        _, model = ResCapsuleNetworkWith3LevelV1(name=f'res_capsnet_3level_v1_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    elif args.model == 'res_capsnet_3level_v2':
        _, model = ResCapsuleNetworkWith3LevelV2(name=f'res_capsnet_3level_v2_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    elif args.model == 'res_capsnet_3level_v3':
        _, model = ResCapsuleNetworkWith3LevelV3(name=f'res_capsnet_3level_v2_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    elif args.model == 'res50_capsnet_3level':
        _, model = Resnet50WithCapsuleNetworkWith3Level(name=f'res50_capsnet_3level_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    else:
        _, model = CapsuleNetworkWith3Level(name=f'capsnet_3level_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)

    model.load_weights(os.path.join(args.save_dir, args.path_model))
    y_pred, x_recon = model.predict(x_test, batch_size=100)

    # MNIST
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    figure = utils.plot_confusion_matrix(cm, class_names, title='Матрица ошибок (трехуровневая капсульная сеть)', show=True)
    figure.savefig(os.path.join(args.save_dir, f'confusion_matrix_{args.dataset}.png'))

    report = classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    figure = utils.ClassificationReportPlotWriter.plot(report, title='Отчет классификации (трехуровневая капсульная сеть)', show=True)
    figure.savefig(os.path.join(args.save_dir, f'classification_report_{args.dataset}.png'))

    img = utils.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(args.save_dir, f'real_and_recon_{args.dataset}.png'))
    plt.imshow(plt.imread(os.path.join(args.save_dir, f'real_and_recon_{args.dataset}.png')))
    plt.show()

    utils.plot_log(os.path.join(args.save_dir, f'history_training_{args.dataset}.csv'),
                  'length_accuracy', 'val_length_accuracy',
                  'Точность (accuracy) при обучении', 'Точность (accuracy) при валидации',
                  'Значения метрики точности (accuracy) при обучении и при валидации',
                   color='b', show=True, save_dir=args.save_dir)

    utils.plot_log(os.path.join(args.save_dir, f'history_training_{args.dataset}.csv'),
                  'length_loss', 'val_length_loss',
                  'Потери (losses) при обучении', 'Потери (losses) при валидации',
                  'Значения метрики потери (loss) при обучении и при валидации',
                   color='b', show=True, save_dir=args.save_dir)
