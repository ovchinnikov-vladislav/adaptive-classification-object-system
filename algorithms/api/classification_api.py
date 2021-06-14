from flask import Blueprint, request
import numpy as np
from .service.classification_service import classification_video

classification_api = Blueprint('classification_api', __name__)


@classification_api.route('/video_classification', methods=['POST'])
def video_classification():
    file = request.files['file']
    npzfile = np.load(file)
    print(npzfile['arr_0'].shape)
    # arr = np.fromstring(data['video'], dtype=float).reshape((8, 112, 112, 3))
    # print(arr.shape)
    #
    # return classification_video(arr)
