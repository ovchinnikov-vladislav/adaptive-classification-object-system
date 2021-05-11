import os


yolo_v3_weights = os.path.abspath('./model_data/yolov3.tf')
yolo_v3_tiny_weights = os.path.abspath('./model_data/yolov3-tiny.tf')
yolo_v4_weights = os.path.abspath('./model_data/yolov4.tf')
caps_yolo_weights = os.path.abspath('...')

yolo_v3_anchors = os.path.abspath('./model_data/yolo3_anchors.txt')
yolo_v3_tiny_anchors = os.path.abspath('./model_data/yolo3_tiny_anchors.txt')
yolo_v4_anchors = os.path.abspath('./model_data/yolo4_anchors.txt')

coco_classes_ru = os.path.abspath('./model_data/coco_classes_ru.txt')
coco_classes_en = os.path.abspath('./model_data/coco_classes_en.txt')

deepsort_model = os.path.abspath('./model_data/deepsort.pb')
deepsort_caps_model = os.path.abspath('...')

font_cv = os.path.abspath('./font/FiraMono-Medium.otf')

rabbitmq_host = 'amqp://guest:guest@192.168.0.13:5672/'