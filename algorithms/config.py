import os

path = os.path.dirname(os.path.abspath(__file__))

yolo_v3_weights = os.path.join(path, 'resources', 'data', 'yolov3.tf')
yolo_v3_person_weights = os.path.join(path, 'resources', 'data', 'yolov3-person.tf')
yolo_v3_tiny_weights = os.path.join(path, 'resources', 'data', 'yolov3-tiny.tf')
yolo_v4_weights = os.path.join(path, 'resources', 'data', 'yolov4.tf')
yolo_caps_weights = os.path.join(path, 'resources', 'data', '...')

yolo_v3_anchors = os.path.join(path, 'resources', 'data', 'yolo3_anchors.txt')
yolo_v3_tiny_anchors = os.path.join(path, 'resources', 'data', 'yolo3_tiny_anchors.txt')
yolo_v4_anchors = os.path.join(path, 'resources', 'data', 'yolo4_anchors.txt')
yolo_caps_anchors = os.path.join(path, 'resources', 'data', 'yolo_caps_anchors.txt')

coco_classes_ru = os.path.join(path, 'resources', 'data', 'coco_classes_ru.txt')
coco_classes_en = os.path.join(path, 'resources', 'data', 'coco_classes_en.txt')

deepsort_model = os.path.join(path, 'resources', 'data', 'deepsort.pb')
deepsort_caps_model = os.path.join(path, 'resources', 'data', '...')
mars_datasets = os.path.join(path, 'resources', 'data', 'mars')

ucf24_caps_model = os.path.join(path, 'resources', 'data', 'ucf24-caps.tf')
ucf24_classes_ru = os.path.join(path, 'resources', 'data', 'ucf24_classes_ru.txt')
ucf24_classes_en = os.path.join(path, 'resources', 'data', 'ucf24_classes_en.txt')

font_cv = os.path.join(path, 'resources', 'font', 'FiraMono-Medium.otf')

rabbitmq_host = 'amqp://guest:guest@192.168.0.13:5672/'
