from flask import Blueprint, request
import numpy as np
from .service.classification_service import classification_video

classification_api = Blueprint('classification_api', __name__)


@classification_api.route('/video_classification', methods=['POST'])
def video_classification():
    data = request.json
    arr = np.array(data['video'])
    print(arr.shape)

    return classification_video(arr)
