from flask_ngrok import run_with_ngrok
from flask import Flask


if __name__ == '__main__':
    from api import detection_api

    app = Flask(__name__)
    app.register_blueprint(detection_api.detection_api)
    run_with_ngrok(app)

    app.run()
