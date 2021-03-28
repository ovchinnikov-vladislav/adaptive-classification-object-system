import queue
import cv2
import numpy as np
import time
from libs.yolo3.model import YoloModel
import threading
from urllib import request
from PIL import Image


inputQ = queue.Queue()
outputQ = queue.Queue()


def thread_input_image(video_path):
    while True:
        im = Image.open(request.urlopen(video_path))
        inputQ.put(np.array(im))


def thread_process_image(yolo):
    while True:
        if inputQ.qsize() >= 50:
            while not inputQ.empty():
                image = inputQ.get()
                img, info_det = yolo.detect_image(np.array(image))
                outputQ.put(img)


if __name__ == '__main__':
    yolo = YoloModel()
    inputThread = threading.Thread(target=thread_input_image, args=('http://192.168.0.16:8080/shot.jpg', ))
    inputThread.start()
    processThread = threading.Thread(target=thread_process_image, args=(yolo, ))
    processThread.start()
    while True:
        if outputQ.qsize() >= 100:
            while not outputQ.empty():
                image = outputQ.get()
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", image)
                if cv2.waitKey(1) == 27:
                    break
                time.sleep(0.02)


