import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from libs.capsnets import losses
from libs.capsnets.models.rescaps import capsnet_3level
from libs import utls
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--routings', default=3, type=int)
parser.add_argument('--save_dir', default='./')
parser.add_argument('--dataset', default='mnist', help='value: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.9, type=float)
parser.add_argument('--lam_recon', default=0.392, type=float)

if __name__ == '__main__':
    # CapsNet Mnist
    args = parser.parse_args()

    # load data
    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)
    # define model

    # model, eval_model, manipulate_model = caps_net(shape=x_train.shape[1:],
    #                                                num_classes=len(np.unique(np.argmax(y_train, 1))),
    #                                                routings=args.routings)
    #
    # model.summary()
    #
    # # callbacks
    # log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    # tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs')
    # checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
    #                                        save_best_only=True, save_weights_only=True, verbose=1)
    # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    #
    # # compile the model
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #               loss=[losses.margin_loss, 'mse'],
    #               loss_weights=[1., args.lam_recon],
    #               metrics='accuracy')
    #
    # model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
    #           validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    #
    # model.save_weights(f'{args.save_dir}/trained_basic_capsnet_model_{args.dataset}.h5')
    # eval_model.save_weights(f'{args.save_dir}/eval_basic_capsnet_model_{args.dataset}.h5')
    # manipulate_model.save_weights(f'{args.save_dir}/manipulate_basic_capsnet_model_{args.dataset}.h5')
    #
    # print(f'Trained model saved to \'{args.save_dir}/trained_basic_capsnet_model_{args.dataset}.h5\'')
    # print(f'Evaluated model saved to \'{args.save_dir}/eval_basic_capsnet_model_{args.dataset}.h5\'')
    # print(f'Manipulated model saved to \'{args.save_dir}/manipulate_basic_capsnet_model_{args.dataset}.h5\'')
    #
    # utls.plot_log(args.save_dir + '/log.csv', show=True)

    model = capsnet_3level(shape=x_train.shape[1:],
                           num_classes=len(np.unique(np.argmax(y_train, 1))),
                           routings=args.routings)

    model.summary()

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_accuracy',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=losses.margin_loss,
                  metrics='accuracy')

    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(x_test, y_test), callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(f'{args.save_dir}/trained_basic_capsnet_model_{args.dataset}.h5')

    print(f'Trained model saved to \'{args.save_dir}/trained_basic_capsnet_model_{args.dataset}.h5\'')

    utls.plot_log(args.save_dir + '/log.csv', show=True)
