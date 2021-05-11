from flask import Blueprint, Response, request
from api.model.video_camera import YoutubeCamera, VideoCamera
import pika
import json
import config
import logging
from libs.yolo3.model import YoloDetectionModel, YoloTrackingModel

STAT_FANOUT_QUEUE_NAME = "stat.fanout.queue"
STAT_EXCHANGE_NAME = "stat.fanout.exchange"
detection_api = Blueprint('detection_api', __name__)
model = YoloTrackingModel()


def gen(cam, user_id, detection_process_id):
    i = 1
    while True:
        frame, det_info = cam.get_frame()

        try:
            rmq_parameters = pika.URLParameters(config.rabbitmq_host)
            rmq_connection = pika.BlockingConnection(rmq_parameters)
            rmq_channel = rmq_connection.channel()

            for obj in det_info:
                json_str = {
                    'type': 'OBJECT_DETECTION',
                    'attributes':
                        {
                            'clazz': obj.get_class(), 'box': obj.get_box(),
                            'score': obj.get_score(), 'numObject': obj.get_num(),
                            'userId': user_id, 'detectionProcessId': detection_process_id,
                            'iteration': i, 'image': obj.get_img()
                        }
                }
                json_dumps = json.dumps(json_str)
                rmq_channel.basic_publish(exchange=STAT_EXCHANGE_NAME,
                                          routing_key=STAT_FANOUT_QUEUE_NAME,
                                          body=json_dumps.encode('utf-8'))
            i += 1

            rmq_connection.close()
        except Exception as e:
            logging.error(e)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


@detection_api.route('/init/<model_name>')
def init_model(model_name):
    global model
    if model_name == 'yolov3_tracking':
        model = YoloTrackingModel()


@detection_api.route('/youtube_video_feed/<video_id>/<user_id>/<detection_process_id>')
def youtube_video_feed(video_id, user_id, detection_process_id):
    model.clear_tracker()
    camera = YoutubeCamera(model, video_id)
    return Response(gen(camera, user_id, detection_process_id), mimetype='multipart/x-mixed-replace;boundary=frame')


@detection_api.route('/camera_video_feed/<user_id>/<detection_process_id>')
def camera_video_feed(user_id, detection_process_id):
    model.clear_tracker()
    video_addr = request.args.get("video_addr")
    camera = VideoCamera(model, video_addr)
    return Response(gen(camera, user_id, detection_process_id), mimetype='multipart/x-mixed-replace;boundary=frame')
