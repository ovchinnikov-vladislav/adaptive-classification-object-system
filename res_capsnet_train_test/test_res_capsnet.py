from libs import utls
from libs.capsnets.models.rescaps import ResCapsuleNetworkV1, ResCapsuleNetworkV2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--routings', default=3)
parser.add_argument('--save_dir', default='res_capsnet_v1_logs')
parser.add_argument('--dataset', default='mnist', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--model', default='res_capsnet_v2', help='values: res_capsnet_v1, res_capsnet_v2')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    if args.model == 'res_capsnet_v2':
        _, model = ResCapsuleNetworkV2(name=f'res_capsnet_v2_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)
    else:
        _, model = ResCapsuleNetworkV1(name=f'res_capsnet_v1_{args.dataset}') \
            .create(input_shape=x_train.shape[1:],
                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                    routings=args.routings)

    model.load_weights(
        os.path.join(args.save_dir,
                     f'res_capsnet_v1_{args.dataset}-result-2021-04-18-7466ebc7-bfbe-4d56-86d0-41f04924d877.h5'))

    y_pred, x_recon = model.predict(x_test, batch_size=100)

    utls.plot_log(os.path.join(args.save_dir, 'history_training.csv'), 'length_accuracy', 'val_length_accuracy',
                  'Точность (accuracy) при обучении', 'Точность (accuracy) при валидации',
                  'Значения метрики точности (accuracy) при обучении и при валидации',
                  color='b', show=True, save_dir=args.save_dir)

    utls.plot_log(os.path.join(args.save_dir, 'history_training.csv'), 'length_loss', 'val_length_loss',
                  'Потери (losses) при обучении', 'Потери (losses) при валидации',
                  'Значения метрики потери (loss) при обучении и при валидации',
                  color='b', show=True, save_dir=args.save_dir)

    # MNIST
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    figure = utls.plot_confusion_matrix(cm, class_names, show=True)
    figure.savefig(os.path.join(args.save_dir, "classification_matrix.png"))

    report = classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    figure = utls.ClassificationReportPlotWriter.plot(report, show=True)
    figure.savefig(os.path.join(args.save_dir, "classification_report.png"))

    img = utls.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(args.save_dir, "real_and_recon.png"))
    plt.imshow(plt.imread(os.path.join(args.save_dir, 'real_and_recon.png')))
    plt.show()
