from libs.datasets import coco, wider_faces


if __name__ == '__main__':
    ann_train_path, ann_test_path, ann_val_path = coco.coco_dataset_annotations(['person'], '.',  True, True)
