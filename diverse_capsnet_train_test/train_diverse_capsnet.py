from libs import utls
from libs.capsnets.models.diverse import DiverseCapsuleNetwork
from tensorflow.keras import optimizers
from libs.capsnets.losses import margin_loss
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--shift_fraction', default=0.1, type=float)
parser.add_argument('--dataset', default='mnist', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--lr_decay', default=0.90, type=float)
parser.add_argument('-r', '--routings', default=3, type=int)
parser.add_argument('--save_dir', default='diverse_capsnet_logs')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    builder = DiverseCapsuleNetwork(name=f'diverse_capsule_networks_{args.dataset}')
    model, _ = builder.build(input_shape=x_train.shape[1:],
                             num_classes=len(np.unique(np.argmax(y_train, 1))),
                             routings=args.routings)
    builder.compile(optimizer=optimizers.Adam(lr=args.lr),
                    loss=margin_loss,
                    loss_weights=[1.],
                    metrics=['accuracy'])

    builder.fit([x_train, y_train], y_train, args.batch_size, args.epochs, checkpoint_monitor='val_accuracy',
                validation_data=[[x_test, y_test], y_test], log_dir='diverse_capsnet_logs')
