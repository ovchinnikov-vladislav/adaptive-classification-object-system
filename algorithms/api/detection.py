from flask import Blueprint, Response
from api.model.video_camera import YoutubeCamera
import requests
import json
import uuid

detection_api = Blueprint('detection_api', __name__)


def gen(cam):
    i = 1
    while True:
        frame, det_info = cam.get_frame()
        # for obj in det_info:
        #     try:
        #         headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        #         requests.post('http://localhost:8082/classification-data/' +
        #                       str(uuid.uuid4()) + '/' + str(i),
        #                       json={'clazz': obj.get_class(), 'box': obj.get_box(),
        #                             'score': obj.get_score(), 'num': obj.get_num()},
        #                       headers=headers)
        #     except:
        #         pass
        # i += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@detection_api.route('/video_feed/<video_id>')
def video_feed(video_id):
    camera = YoutubeCamera(video_id)
    return Response(gen(camera), mimetype='multipart/x-mixed-replace;boundary=frame')
