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

STAT_FANOUT_QUEUE_NAME = "stat.fanout.queue"
STAT_EXCHANGE_NAME = "stat.fanout.exchange"


tracking_model = ObjectDetectionModel(model='yolo_caps', classes=['person', 'face'], use_tracking=True)
video_model = VideoClassCapsNetModel()
# tracking_model = None

def event_classification(objects_frames, objects_classes):
    model = VideoClassCapsNetModel()
    while True:
        for key in objects_frames.keys():
            value = objects_frames.get(key)
            if len(value) >= 20:
                frames_numpy = np.stack(value, axis=0)
                num_frames, height, width, _ = frames_numpy.shape

                # video_io.vwrite('video.avi', frames_numpy)
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # video = cv2.VideoWriter('video.avi', fourcc, 1, (width, height))
                #
                # for f in np.split(frames_numpy, num_frames, axis=0):
                #     f = np.squeeze(f)
                #     video.write(f)
                #
                # video.release()
                result = model.predict(frames_numpy)
                objects_classes[key] = result
        time.sleep(1 / 60)


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

                    if config.video_classification_input_queue.empty() and len(frames) == 8:
                        config.video_classification_input_queue.put(objects_frames)
                    if not config.video_classification_output_queue.empty():
                        objects_classes = config.video_classification_output_queue.get()

                    if objects_classes.get(obj.get_num(), None) is not None:
                        # print(objects_classes.get(obj.get_num(), None))
                        objects_classes[obj.get_num()] = None
                        objects_frames[obj.get_num()] = []

                    # if len(objects_frames[obj.get_num()]) > 25:
                    #     video = np.stack(frames, axis=0)
                    #     event_class = video_model.predict(video)
                    #     object_classes[obj.get_num()] = event_class
                    #     objects_frames[obj.get_num()] = frames
                    #
                    # if object_classes.get(obj.get_num(), None) is not None:
                    #     print(object_classes.get(obj.get_num(), None))

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
