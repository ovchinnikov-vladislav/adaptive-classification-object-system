from flask import Response
import logging
from api.model.video_camera import YoutubeCamera, VideoCamera
import json
import config
from libs.detection.utils import ObjectDetectionModel
from libs.capsnets.utils import VideoClassCapsNetModel
import cv2
import numpy as np
import base64
import time
from threading import Thread
import requests

STAT_FANOUT_QUEUE_NAME = "stat.fanout.queue"
STAT_EXCHANGE_NAME = "stat.fanout.exchange"


tracking_model = ObjectDetectionModel(model='yolo_caps', classes=['person', 'face'], use_tracking=True)


def get_video_frame_with_tracking(cam, user_id, tracking_process_id):
    i = 1

    objects_frames = dict()
    objects_classes = dict()

    # video_event_classification = Thread(target=event_classification, args=(objects_frames, objects_classes))
    # video_event_classification.start()
    while True:
        frame, det_info = cam.get_frame()

        try:
            # rmq_parameters = pika.URLParameters(config.rabbitmq_addr)
            # rmq_connection = pika.BlockingConnection(rmq_parameters)
            # rmq_channel = rmq_connection.channel()

            for obj in det_info:
                if obj.get_class() == 'person':
                    imgdata = base64.b64decode(obj.get_img())
                    decoded = np.frombuffer(imgdata, np.uint8)
                    decoded = cv2.imdecode(decoded, cv2.IMREAD_COLOR)

                    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnt = contours[0]
                    x, y, w, h = cv2.boundingRect(cnt)
                    decoded = decoded[y:y + h, x:x + w]

                    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    decoded = cv2.resize(decoded, (240, 320))

                    frames = objects_frames.get(obj.get_num(), [])
                    frames.append(decoded)
                    objects_frames[obj.get_num()] = frames

                    if len(objects_frames[obj.get_num()]) > 8:
                        objects_frames[obj.get_num()] = []

                    if config.video_classification_input_queue.empty():
                        config.video_classification_input_queue.put(objects_frames)

                    if not config.video_classification_output_queue.empty():
                        objects_classes = config.video_classification_output_queue.get()

                    if objects_classes.get(obj.get_num(), None) is not None:
                        print(objects_classes.get(obj.get_num()))

                    #
                    # json_str = {
                    #     'type': 'OBJECT_DETECTION',
                    #     'attributes':
                    #         {
                    #             'clazz': obj.get_class(), 'box': obj.get_box(),
                    #             'score': obj.get_score(), 'numObject': obj.get_num(),
                    #             'userId': user_id, 'detectionProcessId': tracking_process_id,
                    #             'iteration': i, 'image': obj.get_img()
                    #         }
                    # }
                    #
                    # json_dumps = json.dumps(json_str)
                    # rmq_channel.basic_publish(exchange=STAT_EXCHANGE_NAME,
                    #                           routing_key=STAT_FANOUT_QUEUE_NAME,
                    #                           body=json_dumps.encode('utf-8'))
            i += 1

            # rmq_connection.close()
        except Exception as e:
            logging.error(e)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')


def get_video_with_tracking_objects(video_id, user_id, tracking_process_id, type='youtube'):
    tracking_model.clear_tracker()

    if type == 'youtube':
        camera = YoutubeCamera(tracking_model, video_id)
    elif type == 'video':
        camera = VideoCamera(tracking_model, video_id)
    else:
        raise Exception('error detection objects')
    return Response(get_video_frame_with_tracking(camera, user_id, tracking_process_id),
                    mimetype='multipart/x-mixed-replace;boundary=frame')
