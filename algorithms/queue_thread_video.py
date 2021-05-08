import queue
import cv2
import numpy as np
import time
from libs.yolo3.model import YoloDetectionModel
import threading
from urllib import request
from PIL import Image


q = queue.Queue()


def thread_input_image(video_path, yolo):
    while True:
        im = Image.open(request.urlopen(video_path))
        img, info_det = yolo.detect_image(im)
        print(info_det)
        q.put(img)
      #  print('input: ', q.qsize())


def thread_output_image():
    while True:
        if q.qsize() >= 400:
            while not q.empty():
                image = q.get()
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", image)
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.02)


if __name__ == '__main__':
    yolo = YoloDetectionModel()
    inputThread = threading.Thread(target=thread_input_image, args=('http://192.168.0.16:8080/shot.jpg', yolo))
    inputThread.start()
    outputThread = threading.Thread(target=thread_output_image)
    outputThread.start()


