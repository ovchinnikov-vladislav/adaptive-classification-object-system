from flask import Blueprint, request
import numpy as np
from .service.classification_service import classification_video

classification_api = Blueprint('classification_api', __name__)


@classification_api.route('/video_classification', methods=['POST'])
def video_classification():
    file = request.files['file']
    npzfile = np.load(file)
    return classification_video(npzfile['arr_0'])
