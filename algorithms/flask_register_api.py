from flask_ngrok import run_with_ngrok
from flask import Flask
from api import detection
from api.classification import classification_api
from api.training import training_api


if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(detection.detection_api)
    app.register_blueprint(classification_api)
    app.register_blueprint(training_api)
    run_with_ngrok(app)

    app.run()
