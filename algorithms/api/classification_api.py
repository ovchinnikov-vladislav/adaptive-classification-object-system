from flask import Blueprint, request
import numpy as np
from .service.classification_service import videos_classification

classification_api = Blueprint('classification_api', __name__)


@classification_api.route('/video_classification', methods=['POST'])
def video_classification():
    file = request.files['file']
    npzfile = np.load(file)

    return videos_classification(npzfile)
