from libs.capsnets.utils import VideoClassCapsNetModel

video_model = VideoClassCapsNetModel()


def videos_classification(npzfile):
    result = dict()
    for key in npzfile.files:
        result[key] = video_model.predict_short(npzfile[key])

    return result
