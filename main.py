import numpy as np
from tensorflow.keras import models, optimizers
from PIL import Image
from bmstu.capsnet import models, losses
from bmstu import utls
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks


# def train(model,  # type: models.Model
#           data, args):
#     """
#     Training a CapsuleNet
#     :param model: the CapsuleNet model
#     :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
#     :param args: arguments
#     :return: The trained model
#     """
#     # unpacking the data
#     (x_train, y_train), (x_test, y_test) = data
#
#     # callbacks
#     log = callbacks.CSVLogger(args.save_dir + '/log.csv')
#     checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
#                                            save_best_only=True, save_weights_only=True, verbose=1)
#     lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
#
#     # compile the model
#     model.compile(optimizer=optimizers.Adam(lr=args.lr),
#                   loss=[losses.margin_loss, 'mse'],
#                   loss_weights=[1., args.lam_recon],
#                   metrics='accuracy')
#
#     """
#     # Training without data augmentation:
#     model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
#               validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
#     """
#
#     # Begin: Training with data augmentation ---------------------------------------------------------------------#
#     def train_generator(x, y, batch_size, shift_fraction=0.):
#         train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
#                                            height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
#         generator = train_datagen.flow(x, y, batch_size=batch_size)
#         while 1:
#             x_batch, y_batch = generator.next()
#             yield (x_batch, y_batch), (y_batch, x_batch)
#
#     # Training with data augmentation. If shift_fraction=0., no augmentation.
#     model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
#               steps_per_epoch=int(y_train.shape[0] / args.batch_size),
#               epochs=args.epochs,
#               validation_data=((x_test, y_test), (y_test, x_test)), batch_size=args.batch_size)
#     # End: Training with data augmentation -----------------------------------------------------------------------#
#
#     model.save_weights(args.save_dir + '/trained_model.h5')
#     print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
#
#     plot_log(args.save_dir + '/log.csv', show=True)
#
#     return model
#
#
# def test(model, data, args):
#     x_test, y_test = data
#     y_pred, x_recon = model.predict(x_test, batch_size=100)
#     print('-' * 30 + 'Begin: test' + '-' * 30)
#     print('y_test.shape[0]', y_test.shape[0])
#     print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
#
#     img = utl.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
#     image = img * 255
#     Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
#     print()
#     print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
#     print('-' * 30 + 'End: test' + '-' * 30)
#     plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
#     plt.show()
#
#
# def manipulate_latent(model, data, args):
#     print('-' * 30 + 'Begin: manipulate' + '-' * 30)
#     x_test, y_test = data
#     index = np.argmax(y_test, 1) == args.digit
#     number = np.random.randint(low=0, high=sum(index) - 1)
#     x, y = x_test[index][number], y_test[index][number]
#     x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
#     noise = np.zeros([1, 10, 16])
#     x_recons = []
#     for dim in range(16):
#         for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
#             tmp = np.copy(noise)
#             tmp[:, :, dim] = r
#             x_recon = model.predict([x, y, tmp])
#             x_recons.append(x_recon)
#
#     x_recons = np.concatenate(x_recons)
#
#     img = utl.combine_images(x_recons, height=16)
#     image = img * 255
#     Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
#     print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
#     print('-' * 30 + 'End: manipulate' + '-' * 30)


class Args:
    def __init__(self):
        self.epochs = 2
        self.batch_size = 112
        self.lr = 0.001
        self.lr_decay = 0.9
        self.lam_recon = 0.392
        self.routings = 3
        self.shift_fraction = 0.1
        self.save_dir = '.'
        self.digit = 5


args = Args()

# load data
(x_train, y_train), (x_test, y_test) = utls.load('mnist')
# define model

model, eval_model, manipulate_model = models.CapsNet(shape=x_train.shape[1:],
                                                     classes=len(np.unique(np.argmax(y_train, 1))),
                                                     routings=args.routings).build()

model.summary()

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=[losses.margin_loss, 'mse'],
              metrics='accuracy')

model.fit([x_train, y_train], [y_train, x_train], batch_size=100, epochs=5,
          validation_data=[[x_test, y_test], [y_test, x_test]])

# model.fit(x_train, y_train, validation_data=[x_test, y_test])

#
# train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
# test(model=eval_model, data=(x_test, y_test), args=args)
