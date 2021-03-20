import cv2
from PIL import Image
from bmstu.yolo3.model import YoloModel
import numpy as np
from flask import Flask, render_template, Response, make_response
from urllib import request


class VideoCamera:
    def __init__(self):
        self.yolo = YoloModel()
        # self.video = cv2.VideoCapture('http://192.168.0.16:8080/shot.jpg')

    # def __del__(self):
    #     self.video.release()

    def get_frame(self):
        im = Image.open(request.urlopen('http://192.168.0.16:8080/shot.jpg'))
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        image = self.yolo.detect_image(im)
        result = np.asarray(image)
        ret, jpeg = cv2.imencode('.jpg', result)
        return jpeg.tobytes()


# @app.route('/shot.jpg')
# def shot():
#     frame = camera.get_frame()
#     return Response((b'--frame\r\n'
#                      b'Content-Type: image/jpeg\r\n\r\n' + frame +
#                      b'\r\n\r\n'), mimetype='multipart/x-mixed-replace;'
#                                             'boundary=frame')


if __name__ == '__main__':
    app = Flask(__name__, template_folder='templates/')


    @app.route('/video')
    def video():
        return render_template("index.html")


    @app.route('/video_feed')
    def video_feed():
        return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace;'
                                                     'boundary=frame')


    def gen(cam):
        while True:
            frame = cam.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame +
                   b'\r\n\r\n')


    app.run(host='localhost', port='5000', debug=True)
