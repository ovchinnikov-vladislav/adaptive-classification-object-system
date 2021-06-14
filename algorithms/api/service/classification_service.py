from libs.capsnets.utils import VideoClassCapsNetModel
from datetime import datetime
import json

video_model = VideoClassCapsNetModel()


def videos_classification(npzfile):
    result = dict()
    i = 0
    for key in npzfile.files:
        result[i] = json.dumps(
            {
                "datetime": str(datetime.now()),
                "class": video_model.predict_short(npzfile[key])
            })
        i += 1

    return result
