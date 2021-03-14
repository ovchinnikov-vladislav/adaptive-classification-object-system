import shutil
import os
import tensorflow_datasets as tfds


def _prepare_annotation(filename, path, dataset):
    with open(filename, 'w') as file:
        for example in dataset:
            string = path + example['image/filename'].numpy().decode()
            height, width, _ = example['image'].shape
            bbox = example['faces']['bbox'].numpy()
            bboxs = ''
            for i in range(len(bbox)):
                bbox_string = f'{int(bbox[i][1] * width)},{int(bbox[i][0] * height)},' \
                              f'{int(bbox[i][3] * width)},{int(bbox[i][2] * height)},{0}'
                bboxs += ' ' + bbox_string
            string += bboxs
            print(string)
            file.write(string + '\n')


def wider_dataset_annotations(root_path='./', is_prepare_annotation=True):
    train_path = f'{root_path}wider_face/WIDER_train/images/'
    test_path = f'{root_path}wider_face/WIDER_test/images/'
    val_path = f'{root_path}wider_face/WIDER_val/images/'
    train_ds = tfds.load('wider_face', split='train', data_dir=root_path)
    val_ds = tfds.load('wider_face', split='validation', data_dir=root_path)
    test_ds = tfds.load('wider_face', split='test', data_dir=root_path)

    list_d = os.listdir(os.path.join(root_path, 'downloads', 'extracted'))
    for d in list_d:
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

    ann_train_path = 'model_data/wider_face_train_annotation.txt'
    ann_test_path = 'model_data/wider_face_test_annotation.txt'
    ann_val_path = 'model_data/wider_face_val_annotation.txt'

    if is_prepare_annotation:
        _prepare_annotation(ann_train_path, train_path, train_ds)
        _prepare_annotation(ann_test_path, test_path, test_ds)
        _prepare_annotation(ann_val_path, val_path, val_ds)

    return ann_train_path, ann_test_path, ann_val_path
