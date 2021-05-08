import os


yolo_v3_weights = os.path.abspath('./model_data/yolov3.tf')
yolo_v4_weights = os.path.abspath('./model_data/yolov4.tf')
caps_yolo_weights = os.path.abspath('...')

yolo_anchors = os.path.abspath('./model_data/yolo_anchors.txt')
coco_classes_ru = os.path.abspath('./model_data/coco_classes_en.txt')
coco_classes_en = os.path.abspath('./model_data/coco_classes_en.txt')

deepsort_model = os.path.abspath('./model_data/deepsort.pb')
deepsort_caps_model = os.path.abspath('...')

font_cv = os.path.abspath('./font/FiraMono-Medium.otf')
