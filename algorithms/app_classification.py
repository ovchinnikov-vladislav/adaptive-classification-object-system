from flask_ngrok import run_with_ngrok
from flask import Flask
from tensorflow.python.client import device_lib
import tensorflow as tf


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


if __name__ == '__main__':
    print(get_available_devices())
    physical_devices = tf.config.list_physical_devices('GPU')
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    from api import classification_api

    app = Flask(__name__)
    app.register_blueprint(classification_api.classification_api)

    app.run(port=5001)
    run_with_ngrok(app)
