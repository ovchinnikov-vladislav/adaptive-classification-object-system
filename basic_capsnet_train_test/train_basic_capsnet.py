import numpy as np
from tensorflow.keras import optimizers
from libs.capsnets import losses
from libs.capsnets.models.basic import CapsuleNetworkV1, CapsuleNetworkV2
from libs import utls
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--routings', default=3, type=int)
parser.add_argument('--save_dir', default='capsnet_v1_logs')
parser.add_argument('--dataset', default='mnist', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.9, type=float)
parser.add_argument('--lam_recon', default=0.392, type=float)
parser.add_argument('--model', default='capsnet_v1', help='value: capsnet_v1, capsnet_v2')

if __name__ == '__main__':
    # CapsNet Mnist
    args = parser.parse_args()

    # load data
    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)
    # define model

    if args.model == 'capsnet_v2':
        builder = CapsuleNetworkV2(name=f'capsnet_v2_{args.dataset}')
    else:
        builder = CapsuleNetworkV1(name=f'capsnet_v1_{args.dataset}')

    model, _ = builder.build(input_shape=x_train.shape[1:],
                             num_classes=len(np.unique(np.argmax(y_train, 1))),
                             routings=args.routings)
    builder.compile(optimizer=optimizers.Adam(lr=args.lr),
                    loss=[losses.margin_loss, 'mse'],
                    metrics='accuracy')

    history = builder.fit(x_train, y_train, args.batch_size, args.epochs, checkpoint_monitor='val_length_accuracy',
                          validation_data=(x_test, y_test), log_dir=args.save_dir)

