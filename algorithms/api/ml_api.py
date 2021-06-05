from flask import Blueprint, request
from .service.ml_service import get_video_with_detection_objects

detection_api = Blueprint('detection_api', __name__)


@detection_api.route('/youtube_video/<video_id>/<user_id>/<detection_process_id>')
def youtube_video(video_id, user_id, detection_process_id):
    return get_video_with_detection_objects(video_id, user_id, detection_process_id, type='youtube')


@detection_api.route('/video_camera/<user_id>/<detection_process_id>')
def camera_video(user_id, detection_process_id):
    video_addr = request.args.get("video_addr")
    return get_video_with_detection_objects(video_addr, user_id, detection_process_id, type='video')
