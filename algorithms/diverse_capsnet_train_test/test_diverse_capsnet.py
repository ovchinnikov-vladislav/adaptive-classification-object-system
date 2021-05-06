from libs import utls
from libs.capsnets.models.diverse import DiverseCapsuleNetwork
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--path_model', default='diverse_capsnet-result-2021-04-17-... .h5')
parser.add_argument('-r', '--routings', default=3)
parser.add_argument('--save_dir', default='capsnet_logs')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    builder = DiverseCapsuleNetwork(name='diverse_capsules')
    _, model = builder.create(input_shape=x_train.shape[1:],
                              num_classes=len(np.unique(np.argmax(y_train, 1))),
                              routings=args.routings)

    model.load_weights(
        os.path.join(args.save_dir, args.path_model))

    y_pred = model.predict(x_test, batch_size=32)

    # MNIST
    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    figure = utls.plot_confusion_matrix(cm, class_names, show=True)
    figure.savefig(os.path.join(args.save_dir, f'classification_matrix_{args.dataset}.png'))

    report = classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    figure = utls.ClassificationReportPlotWriter.plot(report, show=True)
    figure.savefig(os.path.join(args.save_dir, f'classification_report_{args.dataset}.png'))

    utls.plot_log(os.path.join(args.save_dir, f'history_training_{args.dataset}.csv'), 'acc', 'val_acc',
                  'Точность (accuracy) при обучении', 'Точность (accuracy) при валидации',
                  'Значения метрики точности (accuracy) при обучении и при валидации',
                  color='b', show=True, save_dir=args.save_dir)

    utls.plot_log(os.path.join(args.save_dir, f'history_training_{args.dataset}.csv'), 'loss', 'val_loss',
                  'Потери (losses) при обучении', 'Потери (losses) при валидации',
                  'Значения метрики потери (loss) при обучении и при валидации',
                  color='b', show=True, save_dir=args.save_dir)
