from flask_ngrok import run_with_ngrok
from flask import Flask
from api import ml_api


if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(ml_api.detection_api)
    run_with_ngrok(app)

    app.run()
