import cv2
from PIL import Image
from bmstu.yolo3.model import YoloModel
import numpy as np
from flask import Flask, render_template, Response, make_response


class VideoCamera:
    def __init__(self):
        self.yolo = YoloModel()
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        return_value, frame = self.video.read()
        image = Image.fromarray(frame)
        image = self.yolo.detect_image(image)
        result = np.asarray(image)
        ret, jpeg = cv2.imencode('.jpg', result)
        return jpeg.tobytes()


camera = None
yolo = None
app = Flask(__name__, template_folder='templates/')


# @app.route('/shot.jpg')
# def shot():
#     frame = camera.get_frame()
#     return Response((b'--frame\r\n'
#                      b'Content-Type: image/jpeg\r\n\r\n' + frame +
#                      b'\r\n\r\n'), mimetype='multipart/x-mixed-replace;'
#                                             'boundary=frame')


@app.route('/video')
def video():
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


if __name__ == '__main__':
    camera = VideoCamera()
    app.run(host='localhost', port='5000', debug=True)
