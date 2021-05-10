from flask import Blueprint, Response, request
from api.model.video_camera import YoutubeCamera, VideoCamera
import requests

detection_api = Blueprint('detection_api', __name__)


def gen(cam, user_id, detection_process_id):
    i = 1
    while True:
        frame, det_info = cam.get_frame()
        for obj in det_info:
            headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
            requests.post('http://localhost:8082/detection-objects/' +
                          str(user_id) + '/' + str(detection_process_id),
                          json={'clazz': obj.get_class(), 'box': obj.get_box(),
                                'score': obj.get_score(), 'numObject': obj.get_num(),
                                'userId': user_id, 'detectionProcessId': detection_process_id,
                                'iteration': i, 'image': obj.get_img()},
                          headers=headers)
        i += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@detection_api.route('/youtube_video_feed/<video_id>/<user_id>/<detection_process_id>')
def youtube_video_feed(video_id, user_id, detection_process_id):
    camera = YoutubeCamera(video_id)
    return Response(gen(camera, user_id, detection_process_id), mimetype='multipart/x-mixed-replace;boundary=frame')


@detection_api.route('/camera_video_feed/<user_id>/<detection_process_id>')
def camera_video_feed(user_id, detection_process_id):
    video_addr = request.args.get("video_addr")
    camera = VideoCamera(video_addr)
    return Response(gen(camera, user_id, detection_process_id), mimetype='multipart/x-mixed-replace;boundary=frame')
