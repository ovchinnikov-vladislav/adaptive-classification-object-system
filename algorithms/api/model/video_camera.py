import cv2
import numpy as np
import time
import streamlink
from PIL import Image
from urllib import request
import copy
from libs.yolo3.model import YoloDetectionModel, YoloTrackingModel


class YoutubeCamera:
    def __init__(self, video_id):
        self.model = YoloTrackingModel()
        url = f'https://www.youtube.com/watch?v={video_id}'

        streams = streamlink.streams(url)
        self.video = cv2.VideoCapture(streams["720p"].url)
        self.video.set(cv2.CAP_PROP_FPS, 16)

    def get_frame(self):
        return_value, img = self.video.read()
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.model.detect_image(img)
        t2 = time.time()
        print('Общее время детекции: ' + str(int(1000 * (t2 - t1))))
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        return img.tobytes()


class VideoCamera:
    def __init__(self, model, url):
        self.model = model
        self.url = url

    def get_frame(self):
        img = Image.open(request.urlopen(self.url))
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.model.detect_image(img)
        t2 = time.time()
        print('Общее время детекции: ' + str(int(1000 * (t2 - t1))))
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        return img.tobytes()
