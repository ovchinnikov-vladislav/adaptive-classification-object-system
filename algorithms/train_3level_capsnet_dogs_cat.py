import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from libs.capsnets import losses
from libs.capsnets.models.rescaps import capsnet_3level, res_capsnet_3level, res50_capsnet_3level
from libs.capsnets.models.basic import caps_net_without_decoder
from libs import utils
import argparse
import shutil, os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--routings', default=1, type=int)
parser.add_argument('--save_dir', default='./')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--lr_decay', default=0.9, type=float)
parser.add_argument('--lam_recon', default=0.392, type=float)

if __name__ == '__main__':
    # CapsNet Mnist
    args = parser.parse_args()

    # load data
    original_dataset_dir = 'D:\\Downloads\\train\\train'
    base_dir = 'D:\\Downloads\\cats_and_dogs_small'
    # os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    # os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    # os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    # os.mkdir(test_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    # os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    # os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    # os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    # os.mkdir(validation_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    # os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    # os.mkdir(test_dogs_dir)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True, )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                        batch_size=16, classes=['dogs', 'cats'])

    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                            batch_size=16, classes=['dogs', 'cats'])

    model = res50_capsnet_3level(shape=(150, 150, 3), num_classes=2, routings=args.routings)

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

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(train_generator,
                        steps_per_epoch=100,
                        epochs=30,
                        validation_data=validation_generator,
                        callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(f'{args.save_dir}/trained_basic_capsnet_model_dogs_cats.h5')

    print(f'Trained model saved to \'{args.save_dir}/trained_basic_capsnet_model_dogs_cats.h5\'')

    utils.plot_log(args.save_dir + '/log.csv', show=True)
