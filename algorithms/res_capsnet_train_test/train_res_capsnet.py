from libs import utls
from libs.capsnets.models.rescaps import ResCapsuleNetworkV1, ResCapsuleNetworkV2, Resnet50ToCapsuleNetwork
from tensorflow.keras import optimizers
from libs.capsnets.losses import margin_loss
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--shift_fraction', default=0.1, type=float)
parser.add_argument('--dataset', default='mnist', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--lr_decay', default=0.90, type=float)
parser.add_argument('-r', '--routings', default=3, type=int)
parser.add_argument('--save_dir', default='capsnet_logs')
parser.add_argument('--model', default='res_capsnet_v2', help='values: res_capsnet_v1, res_capsnet_v2, '
                                                              'res50_to_capsnet')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    if args.model == 'res_capsnet_v2':
        model = ResCapsuleNetworkV2(name=f'res_capsnet_v2_{args.dataset}')
    elif args.model == 'res50_to_capsnet':
        model = Resnet50ToCapsuleNetwork(name=f'res50_to_capsnet_{args.dataset}')
    else:
        model = ResCapsuleNetworkV1(name=f'res_capsnet_v1_{args.dataset}')

    model.build(input_shape=x_train.shape[1:],
                num_classes=len(np.unique(np.argmax(y_train, 1))),
                routings=args.routings)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  metrics='accuracy')

    history = model.fit(x_train, y_train, args.batch_size, args.epochs, checkpoint_monitor='val_length_accuracy',
                        validation_data=(x_test, y_test), log_dir=args.save_dir)
