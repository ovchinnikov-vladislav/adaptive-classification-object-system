import numpy as np
from tensorflow.keras import optimizers
from libs.capsnets import losses
from libs.capsnets.models.efficient import EfficientCapsuleNetwork
from libs import utls
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--routings', default=3, type=int)
parser.add_argument('--save_dir', default='efficient_capsnet_logs')
parser.add_argument('--dataset', default='cifar10', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.9, type=float)
parser.add_argument('--lam_recon', default=0.392, type=float)

if __name__ == '__main__':
    # CapsNet Mnist
    args = parser.parse_args()

    # load data
    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)
    # define model

    builder = EfficientCapsuleNetwork(name=f'efficient_capsnet_{args.dataset}')

    model, _ = builder.build(input_shape=x_train.shape[1:],
                             num_classes=len(np.unique(np.argmax(y_train, 1))))
    builder.compile(optimizer=optimizers.Adam(lr=5e-4),
                    loss=[losses.margin_loss, 'mse'],
                    loss_weights=[1., 0.392],
                    metrics='accuracy')

    history = builder.fit(x_train, y_train, args.batch_size, args.epochs, checkpoint_monitor='val_length_accuracy',
                          validation_data=(x_test, y_test), log_dir=args.save_dir)
