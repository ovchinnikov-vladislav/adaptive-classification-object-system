import cv2
import numpy as np
import time
import streamlink
from PIL import Image
from urllib import request
from threading import Thread
from libs.detection.utils import analyze_detection_outputs, analyze_tracks_outputs
import matplotlib.pyplot as plt
import config


class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 100)

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
    def __init__(self, model, video_id):
        self.model = model
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
    def __init__(self, model, src):
        self.model = model
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


# class ThreadedCamera:
#     def __init__(self, src):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 100)
#
#         # FPS = 1/X
#         # X = desired FPS
#         self.FPS = 1 / 60
#         self.FPS_MS = int(self.FPS * 1000)
#
#         self.status, self.frame = None, None
#
#         # Start frame retrieval thread
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#
#     def update(self):
#         while True:
#             if self.capture.isOpened():
#                 self.status, self.frame = self.capture.read()
#             time.sleep(self.FPS)
#
#     def get_frame(self):
#         frame = self.frame
#         cv2.waitKey(self.FPS_MS)
#         return frame


# class YoutubeCamera:
#     def __init__(self, model, video_id):
#         self.model = model
#         url = f'https://www.youtube.com/watch?v={video_id}'
#
#         streams = streamlink.streams(url)
#         self.threaded_camera = ThreadedCamera(streams["720p"].url)
#
#     def get_frame(self):
#         img = None
#         while img is None:
#             img = self.threaded_camera.get_frame()
#         img = np.array(img)
#         t1 = time.time()
#         img, det_info = self.model.detect_image(img)
#         t2 = time.time()
#         fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
#         cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.50, color=(255, 0, 0), thickness=2)
#         img = np.asarray(img)
#         ret, img = cv2.imencode('.jpg', img)
#         return img.tobytes(), det_info
#
#
# class VideoCamera:
#     def __init__(self, model, src):
#         self.model = model
#         self.threaded_camera = ThreadedCamera(src)
#         self.params = None
#
#     def get_frame(self):
#         img = None
#         while img is None:
#             img = self.threaded_camera.get_frame()
#         img = np.array(img)
#         t1 = time.time()
#
#         if config.queue_images_for_tracking is not None and config.queue_images_for_tracking.empty():
#             config.queue_images_for_tracking.put(img)
#         if config.queue_tracks_for_tracking is not None and not config.queue_tracks_for_tracking.empty():
#             self.params = config.queue_tracks_for_tracking.get()
#         if self.params is not None:
#             tracks, boxes, scores, classes, nums = self.params
#             cmap = plt.get_cmap('tab20b')
#             colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
#             img, det_info = analyze_detection_outputs(img, (boxes, scores, classes, nums),
#                                                       ['person', 'face'], colors, ignore_classes={'person'})
#             img, det_info = analyze_tracks_outputs(img, tracks, colors)
#
#         t2 = time.time() + 100
#         fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
#         cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#                     fontScale=0.50, color=(255, 0, 0), thickness=2)
#         img = np.asarray(img)
#         ret, img = cv2.imencode('.jpg', img)
#         det_info = []
#         return img.tobytes(), det_info