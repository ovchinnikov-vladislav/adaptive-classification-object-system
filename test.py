from bmstu.yolo3.train import create_model, get_classes, get_anchors
from bmstu.yolo3.model import YoloModel
from PIL import Image

# classes_path = 'model_data/voc_classes.txt'
# anchors_path = 'model_data/yolo_anchors.txt'
#
# class_names = get_classes(classes_path)
# num_classes = len(class_names)
#
# anchors = get_anchors(anchors_path)
#
# model = create_model((416, 416), anchors, num_classes, load_pretrained=False)


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
  #  yolo.close_session()


if __name__ == '__main__':
    yolo = YoloModel()
    detect_img(yolo)
    # model.summary(line_length=150)
