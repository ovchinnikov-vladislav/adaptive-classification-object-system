import shutil
import os
import tensorflow_datasets as tfds


def _prepare_annotation(filename, path, dataset):
    with open(filename, 'w') as file:
        print('Start prepare annotation')
        for example in dataset:
            string = os.path.join(path, example['image/filename'].numpy().decode())
            height, width, _ = example['image'].shape
            bbox = example['faces']['bbox'].numpy()
            bboxs = ''
            for i in range(len(bbox)):
                bbox_string = f'{int(bbox[i][1] * width)},{int(bbox[i][0] * height)},' \
                              f'{int(bbox[i][3] * width)},{int(bbox[i][2] * height)},{0}'
                bboxs += ' ' + bbox_string
            string += bboxs
            file.write(string + '\n')
        print('End prepare annotation')


def wider_dataset_annotations(root_path='./', download=True, is_prepare_annotation=True):
    train_path = os.path.join(root_path, 'wider_face', 'WIDER_train', 'images')
    test_path = os.path.join(root_path, 'wider_face', 'WIDER_test', 'images')
    val_path = os.path.join(root_path, 'wider_face', 'WIDER_val', 'images')

    ann_train_path = './model_data/wider_face_train_annotation.txt'
    ann_test_path = './model_data/wider_face_test_annotation.txt'
    ann_val_path = './model_data/wider_face_val_annotation.txt'

    if download:
        train_ds = tfds.load('wider_face', split='train', data_dir=root_path)
        val_ds = tfds.load('wider_face', split='validation', data_dir=root_path)
        test_ds = tfds.load('wider_face', split='test', data_dir=root_path)

        shutil.rmtree(os.path.join(root_path, 'wider_face'))
        os.mkdir(os.path.join(root_path, 'wider_face'))

        downloads_dir = os.listdir(os.path.join(root_path, 'downloads'))
        for ext_d in downloads_dir:
            if ext_d == 'extracted':
                extracted_dir = os.listdir(os.path.join(root_path, 'downloads', ext_d))
                for d in extracted_dir:
                    list_dir = os.listdir(os.path.join(root_path, 'downloads', 'extracted', d))
                    if len(list_dir) != 0:
                        if list_dir[0] == 'WIDER_train':
                            shutil.move(os.path.join(root_path, 'downloads', 'extracted', d, 'WIDER_train'),
                                        os.path.join(root_path, 'wider_face', 'WIDER_train'))
                        elif list_dir[0] == 'WIDER_test':
                            shutil.move(os.path.join(root_path, 'downloads', 'extracted', d, 'WIDER_test'),
                                        os.path.join(root_path, 'wider_face', 'WIDER_test'))
                        elif list_dir[0] == 'WIDER_val':
                            shutil.move(os.path.join(root_path, 'downloads', 'extracted', d, 'WIDER_val'),
                                        os.path.join(root_path, 'wider_face', 'WIDER_val'))
        if is_prepare_annotation:
            _prepare_annotation(ann_train_path, train_path, train_ds)
            _prepare_annotation(ann_test_path, test_path, test_ds)
            _prepare_annotation(ann_val_path, val_path, val_ds)

    return ann_train_path, ann_test_path, ann_val_path
