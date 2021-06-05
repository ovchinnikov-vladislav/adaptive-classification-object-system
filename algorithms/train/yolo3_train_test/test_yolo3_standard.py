from libs.detection.utils import ObjectDetectionModel
from PIL import Image


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
  #  object_detection_model.close_session()


if __name__ == '__main__':
    yolo = ObjectDetectionModel()
    detect_img(yolo)

