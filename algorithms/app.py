from flask_ngrok import run_with_ngrok
from flask import Flask
from tensorflow.python.client import device_lib
from libs.capsnets.utils import VideoClassCapsNetModel
import config
from multiprocessing import Queue, Process
import numpy as np
import tensorflow as tf


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def predictions_tracks(queue_input, queue_output):
    model = VideoClassCapsNetModel()
    while True:
        if not queue_input.empty():
            object_frames = queue_input.get()
            object_classes = dict()
            for key in object_frames.keys():
                value = object_frames[key]
                if len(value) > 25:
                    video = np.stack(value, axis=0)
                    event_class = model.predict(video)
                    object_classes[key] = event_class
                    if not queue_output.empty():
                        queue_output.get()
                    queue_output.put(object_classes)


if __name__ == '__main__':
    print(get_available_devices())
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    from api import ml_api

    # config.video_classification_input_queue = Queue()
    # config.video_classification_output_queue = Queue()
    #
    # video_classification_process = Process(target=predictions_tracks,
    #                                        args=(config.video_classification_input_queue,
    #                                              config.video_classification_output_queue))
    # video_classification_process.start()

    app = Flask(__name__)
    app.register_blueprint(ml_api.detection_api)
    run_with_ngrok(app)

    app.run()
