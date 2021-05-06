import cv2
from libs.yolo3.model import YoloModel
import numpy as np
import time
import streamlink


class YoutubeCamera:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(YoutubeCamera, cls).__new__(cls)
            cls.instance.yolo = YoloModel()
        return cls.instance

    def set_video_path(self, video_id):
        url = f'https://www.youtube.com/watch?v={video_id}'

        streams = streamlink.streams(url)
        self.video = cv2.VideoCapture(streams["720p"].url)
        self.video.set(cv2.CAP_PROP_FPS, 16)

    def get_frame(self):
        # img = Image.open(request.urlopen('http://192.168.0.16:8080/shot.jpg'))
        return_value, img = self.video.read()
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.yolo.detect_image(img)
        t2 = time.time()
        print('Общее время детекции: ' + str(int(1000 * (t2 - t1))))
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        # img = cv2.resize(img, (640, 480))
        return img.tobytes()
