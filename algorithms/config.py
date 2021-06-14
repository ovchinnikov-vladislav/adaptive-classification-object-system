import os

path = os.path.dirname(os.path.abspath(__file__))

object_detection_weights = os.path.join(path, 'resources', 'data', 'yolo_caps.tf')

yolo_v3_anchors = os.path.join(path, 'resources', 'data', 'yolo3_anchors.txt')
yolo_v3_tiny_anchors = os.path.join(path, 'resources', 'data', 'yolo3_tiny_anchors.txt')
yolo_v4_anchors = os.path.join(path, 'resources', 'data', 'yolo4_anchors.txt')

coco_classes_ru = os.path.join(path, 'resources', 'data', 'coco_classes_ru.txt')
coco_classes_en = os.path.join(path, 'resources', 'data', 'coco_classes_en.txt')

deepsort_model = os.path.join(path, 'resources', 'data', 'deepsort.pb')

ucf24_caps_model = os.path.join(path, 'resources', 'data', 'ucf24-caps.tf')
ucf24_classes_ru = os.path.join(path, 'resources', 'data', 'ucf24_classes_ru.txt')
ucf24_classes_en = os.path.join(path, 'resources', 'data', 'ucf24_classes_en.txt')

font_cv = os.path.join(path, 'resources', 'font', 'FiraMono-Medium.otf')

rabbitmq_addr = os.getenv('RABBITMQ_ADDR', 'amqp://guest:guest@localhost:5672/')

video_classification_addr = os.getenv('VIDEO_CLASSIFICATION_ADDR', 'http://de695089f466.ngrok.io/video_classification')

video_classification_input_queue = None
video_classification_output_queue = None
