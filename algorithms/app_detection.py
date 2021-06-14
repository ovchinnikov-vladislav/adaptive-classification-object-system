from flask_ngrok import run_with_ngrok
from flask import Flask
from tensorflow.python.client import device_lib
import config
from multiprocessing import Queue, Process
import numpy as np
import tensorflow as tf


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


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
                        prediction = json.loads(data)
                        outputs[keys[int(key)]] = f"{prediction['datetime']} {prediction['class']} {prediction['confidence']}"
                        queue_output.put(outputs)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    print(get_available_devices())
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    from api import detection_api

    config.video_classification_input_queue = Queue()
    config.video_classification_output_queue = Queue()

    video_classification_process = Process(target=video_classification, args=(config.video_classification_input_queue,
                                                                              config.video_classification_output_queue))
    video_classification_process.start()

    app = Flask(__name__)
    app.register_blueprint(detection_api.detection_api)
    run_with_ngrok(app)

    app.run()
