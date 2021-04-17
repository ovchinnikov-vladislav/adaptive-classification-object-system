from libs import utls
from libs.capsnets.models.basic import CapsuleNetworkV1
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--routings', default=3)
parser.add_argument('--save_dir', default='capsnet_v1_logs')
parser.add_argument('--dataset', default='mnist', help='value: mnist, fashion_mnist, cifar10, cifar100')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    _, model = CapsuleNetworkV1(name='capsnet_v1').create(input_shape=x_train.shape[1:],
                                                          num_classes=len(np.unique(np.argmax(y_train, 1))),
                                                          routings=args.routings)

    model.load(os.path.join(args.save_dir, f'capsnet_v1-result-2021-04-17.h5'))

    y_pred, x_recon = model.predict(x_test, batch_size=100)

    cm = confusion_matrix(np.argmax(y_test, 1), np.argmax(y_pred, 1))
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    utls.plot_confusion_matrix(cm, class_names)
    print(classification_report(np.argmax(y_test, 1), np.argmax(y_pred, 1)))

    img = utls.combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    plt.imshow(plt.imread(os.path.join(args.save_dir, 'real_and_recon.png')))
    plt.show()
