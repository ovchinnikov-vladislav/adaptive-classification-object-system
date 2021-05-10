import cv2
import numpy as np
import time
import streamlink
from PIL import Image
from urllib import request
from threading import Thread
from libs.yolo3.model import YoloDetectionModel, YoloTrackingModel


class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1 / 60
        self.FPS_MS = int(self.FPS * 1000)

        self.status, self.frame = None, None

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(self.FPS)

    def get_frame(self):
        frame = self.frame
        cv2.waitKey(self.FPS_MS)
        return frame


class YoutubeCamera:
    def __init__(self, video_id):
        self.model = YoloTrackingModel()
        url = f'https://www.youtube.com/watch?v={video_id}'

        streams = streamlink.streams(url)
        self.threaded_camera = ThreadedCamera(streams["720p"].url)

    def get_frame(self):
        img = None
        while img is None:
            img = self.threaded_camera.get_frame()
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.model.detect_image(img)
        t2 = time.time()
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        return img.tobytes(), det_info


class VideoCamera:
    def __init__(self, src):
        self.model = YoloTrackingModel()
        self.threaded_camera = ThreadedCamera(src)

    def get_frame(self):
        img = None
        while img is None:
            img = self.threaded_camera.get_frame()
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.model.detect_image(img)
        t2 = time.time()
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        return img.tobytes(), det_info


class MotionJpegCamera:
    def __init__(self, model, url):
        self.model = model
        self.url = url

    def get_frame(self):
        img = Image.open(request.urlopen(self.url))
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.model.detect_image(img)
        t2 = time.time()
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        return img.tobytes(), det_info
