from libs import utls
from libs.capsnets.models.diverse import DiverseCapsuleNetwork
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='values: mnist, fashion_mnist, cifar10, cifar100')
parser.add_argument('-r', '--routings', default=3)
parser.add_argument('--save_dir', default='./')

if __name__ == '__main__':
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = utls.load(args.dataset)

    builder = DiverseCapsuleNetwork(name='diverse_capsules')
    _, model = builder.create(input_shape=x_train.shape[1:],
                              num_classes=len(np.unique(np.argmax(y_train, 1))),
                              routings=args.routings)

    model.load_weights(f'{args.save_dir}/eval_diverse_capsnet_model_{args.dataset}.h5')

    y_pred = model.predict(x_test, batch_size=64)
    print(y_pred)
    print('-' * 30 + 'Begin: test' + '-' * 30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()
