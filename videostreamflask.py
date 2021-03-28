import cv2
from PIL import Image
from libs.yolo3.model import YoloModel
import numpy as np
from flask import Flask, render_template, Response, make_response
from urllib import request
import time
import streamlink
from flask_ngrok import run_with_ngrok


class VideoCamera:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(VideoCamera, cls).__new__(cls)
            cls.instance.yolo = YoloModel()
        return cls.instance

    def set_video_path(self, video_id):
        url = f'https://www.youtube.com/watch?v={video_id}'

        streams = streamlink.streams(url)
        self.video = cv2.VideoCapture(streams["720p"].url)
        self.video.set(cv2.CAP_PROP_FPS, 24)

    def get_frame(self):
        # img = Image.open(request.urlopen('http://192.168.0.16:8080/shot.jpg'))
        return_value, img = self.video.read()
        img = np.array(img)
        t1 = time.time()
        img, det_info = self.yolo.detect_image(img)
        t2 = time.time()
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        img = np.asarray(img)
        ret, img = cv2.imencode('.jpg', img)
        # img = cv2.resize(img, (640, 480))
        return img.tobytes()


# @app.route('/shot.jpg')
# def shot():
#     frame = camera.get_frame()
#     return Response((b'--frame\r\n'
#                      b'Content-Type: image/jpeg\r\n\r\n' + frame +
#                      b'\r\n\r\n'), mimetype='multipart/x-mixed-replace;'
#                                             'boundary=frame')

camera = VideoCamera()

if __name__ == '__main__':
    app = Flask(__name__, template_folder='templates/')
    run_with_ngrok(app)

    @app.route('/shot.jpg')
    def shot():
        frame = camera.get_frame()
        return Response((b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + frame +
                         b'\r\n\r\n'), mimetype='multipart/x-mixed-replace;'
                                                'boundary=frame')

    @app.route('/video/<video_id>')
    def video(video_id):
        camera.set_video_path(video_id)
        return render_template("index.html")


    @app.route('/video_feed')
    def video_feed():
        return Response(gen(camera), mimetype='multipart/x-mixed-replace;'
                                              'boundary=frame')


    def gen(cam):
        while True:
            frame = cam.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame +
                   b'\r\n\r\n')

    app.run()
