from bmstu import utls
from bmstu.capsnets.models import diverse
from tensorflow.keras import callbacks, optimizers
from bmstu.capsnets.losses import margin_loss
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=25)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--dataset', default='mnist', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.003)
parser.add_argument('--lr_decay', default=0.90)
parser.add_argument('-r', '--routings', default=3)
parser.add_argument('--shift_fraction', default=0.1)
parser.add_argument('--save_dir', default='./')
parser.add_argument('--digit', default=5)


if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load('mnist')

    model, eval_model = diverse.CapsNet(input_shape=x_train.shape[1:],
                                        n_class=len(np.unique(np.argmax(y_train, 1))),
                                        routings=args.routings)
    model.summary()

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=2)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  loss_weights=[1.],
                  metrics={'capsnet': 'accuracy'})

    model.fit([x_train, y_train], y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], y_test], callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(f'{args.save_dir}/trained_diverse_capsnet_model_{args.dataset}.h5')
    eval_model.save_weights(f'{args.save_dir}/eval_diverse_capsnet_model_{args.dataset}.h5')

    print(f'Trained model saved to \'{args.save_dir}/trained_diverse_capsnet_model_{args.dataset}.h5\'')
    print(f'Evaluated model saved to \'{args.save_dir}/eval_diverse_capsnet_model_{args.dataset}.h5\'')

    utls.plot_log(args.save_dir + '/log.csv', show=True)
