from libs.datasets import coco


if __name__ == '__main__':
    ann_train, ann_test, ann_val = coco.coco_dataset_annotations(classes=['person'],
                                                                 root_path='D:/tensorflow_datasets/person')
