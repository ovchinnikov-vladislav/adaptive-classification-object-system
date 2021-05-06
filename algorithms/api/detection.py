from flask import Blueprint, Response
from api.model.video_camera import YoutubeCamera


camera = YoutubeCamera()


detection_api = Blueprint('detection_api', __name__)


@detection_api.route('/detection/shot.jpg')
def shot():
    frame = camera.get_frame()
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame +
                     b'\r\n\r\n'), mimetype='multipart/x-mixed-replace;boundary=frame')


@detection_api.route('/video_feed/<video_id>')
def video_feed(video_id):
    camera.set_video_path(video_id)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace;boundary=frame')


def gen(cam):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')
