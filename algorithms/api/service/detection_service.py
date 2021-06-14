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
from multiprocessing import Process, Queue
import requests

STAT_FANOUT_QUEUE_NAME = "stat.fanout.queue"
STAT_EXCHANGE_NAME = "stat.fanout.exchange"


tracking_model = ObjectDetectionModel(model='yolo3', classes=['person', 'face'], use_tracking=True)


def video_classification(queue_input, queue_output):
    import requests
    import io
    import json
    while True:
        try:
            if not queue_input.empty():
                objects = queue_input.get()
                videos = []
                keys = []
                for key in objects.keys():
                    frames = objects[key]
                    if len(frames) == 8:
                        video = np.stack(frames, axis=0)
                        videos.append(video)
                        keys.append(key)

                if len(videos) > 0 and len(keys) > 0:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, *videos)
                    buf.seek(0)

                    r = requests.post(config.video_classification_addr, files={'file': buf})
                    j = json.loads(r.text)
                    outputs = dict()
                    for key, data in j.items():
                        outputs[keys[int(key)]] = data
                        queue_output.put(outputs)
        except Exception as e:
            print(e)


def get_video_frame_with_tracking(cam, user_id, tracking_process_id):
    i = 1

    objects_frames = dict()
    objects_classes = dict()

    config.video_classification_input_queue = Queue()
    config.video_classification_output_queue = Queue()

    video_classification_process = Process(target=video_classification, args=(config.video_classification_input_queue,
                                                                              config.video_classification_output_queue))
    video_classification_process.start()
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

                    if config.video_classification_input_queue.empty():
                        config.video_classification_input_queue.put(objects_frames)

                    if not config.video_classification_output_queue.empty():
                        objects_classes = config.video_classification_output_queue.get()

                    if objects_classes.get(obj.get_num(), None) is not None \
                            and len(objects_frames[obj.get_num()]) > 8:
                        print(objects_classes.get(obj.get_num()))
                        objects_frames[obj.get_num()] = []
                        objects_classes[obj.get_num()] = None

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
