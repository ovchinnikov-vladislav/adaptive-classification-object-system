import numpy as np
from tensorflow.keras import optimizers
from libs.capsnets.losses import margin_loss
from libs.capsnets.models.complex import (CapsuleNetworkWith3Level, CapsuleNetworkWith4Level,
                                          ResCapsuleNetworkWith3LevelV1, ResCapsuleNetworkWith3LevelV2,
                                          Resnet50WithCapsuleNetworkWith3Level)
from libs import utls
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--routings', default=1, type=int)
parser.add_argument('--save_dir', default='capsnet_3level')
parser.add_argument('--dataset', default='cifar10', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.9, type=float)
parser.add_argument('--lam_recon', default=0.392, type=float)
parser.add_argument('--model', default='capsnet_3level', help='value: capsnet_3level, capsnet_4level, '
                                                              'res_capsnet_3level_v1, res_capsnet_3level_v2, '
                                                              'res50_capsnet_3level')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    if args.model == 'capsnet_4level':
        model = CapsuleNetworkWith4Level(name=f'capsnet_4level_{args.dataset}')
    elif args.model == 'res_capsnet_3level_v1':
        model = ResCapsuleNetworkWith3LevelV1(name=f'res_capsnet_3level_v1_{args.dataset}')
    elif args.model == 'res_capsnet_3level_v2':
        model = ResCapsuleNetworkWith3LevelV2(name=f'res_capsnet_3level_v2_{args.dataset}')
    elif args.model == 'res50_capsnet_3level':
        model = Resnet50WithCapsuleNetworkWith3Level(name=f'res50_capsnet_3level_{args.dataset}')
    else:
        model = CapsuleNetworkWith3Level(name=f'capsnet_3level_{args.dataset}')

    model.build(input_shape=x_train.shape[1:],
                num_classes=len(np.unique(np.argmax(y_train, 1))),
                routings=args.routings)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  metrics='accuracy')

    history = model.fit(x_train, y_train, args.batch_size, args.epochs, checkpoint_monitor='val_length_accuracy',
                        validation_data=(x_test, y_test), log_dir=args.save_dir)
