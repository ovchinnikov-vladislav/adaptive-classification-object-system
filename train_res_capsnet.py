from libs import utls
from libs.capsnets.models import rescaps
from tensorflow.keras import callbacks, optimizers
from libs.capsnets.losses import margin_loss
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--shift_fraction', default=0.1, type=float)
parser.add_argument('--dataset', default='cifar10', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--lr_decay', default=0.90, type=float)
parser.add_argument('-r', '--routings', default=3, type=int)
parser.add_argument('--save_dir', default='./')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    model = rescaps.res_caps_v3_net(shape=x_train.shape[1:],
                                    num_classes=len(np.unique(np.argmax(y_train, 1))),
                                    routings=args.routings)
    model.summary()

    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=2)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  metrics='accuracy')

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              callbacks=[log, tb, checkpoint, lr_decay])

    # # Begin: Training with data augmentation ---------------------------------------------------------------------#
    # def train_generator(x, y, batch_size, shift_fraction=0.):
    #     train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
    #                                        height_shift_range=shift_fraction)
    #     generator = train_datagen.flow(x, y, batch_size=batch_size)
    #     while 1:
    #         x_batch, y_batch = generator.next()
    #         yield [x_batch, y_batch], y_batch
    #
    # # Training with data augmentation. If shift_fraction=0., also no augmentation.
    # model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
    #                     steps_per_epoch=int(y_train.shape[0] / args.batch_size),
    #                     epochs=args.epochs,
    #                     validation_data=[[x_test, y_test], y_test],
    #                     callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(f'{args.save_dir}/trained_diverse_capsnet_model_{args.dataset}.h5')
    # eval_model.save_weights(f'{args.save_dir}/eval_diverse_capsnet_model_{args.dataset}.h5')

    print(f'Trained model saved to \'{args.save_dir}/trained_diverse_capsnet_model_{args.dataset}.h5\'')
    print(f'Evaluated model saved to \'{args.save_dir}/eval_diverse_capsnet_model_{args.dataset}.h5\'')

    utls.plot_log(args.save_dir + '/log.csv', show=True)
