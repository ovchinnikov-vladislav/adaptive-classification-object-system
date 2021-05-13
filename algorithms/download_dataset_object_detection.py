from libs.datasets import coco, wider_faces
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./person', type=str, help='path')


if __name__ == '__main__':
    args = parser.parse_args()
    ann_train_path, ann_test_path, ann_val_path = coco.coco_dataset_annotations(['person'], args.path, True, True)
