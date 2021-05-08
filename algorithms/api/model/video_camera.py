import cv2
import numpy as np
import time
import streamlink
from PIL import Image
from urllib import request


class YoutubeDetectionCamera:
    def __init__(self, model, video_id):
        self.model = model
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


class VideoDetectionCamera:
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


class YoutubeTrackingCamera:
    def __init__(self, detection_model, tracking_model, video_id):
        self.detection_model = detection_model
        self.tracking_model = tracking_model
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