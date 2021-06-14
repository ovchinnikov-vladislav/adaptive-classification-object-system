from libs.capsnets.utils import VideoClassCapsNetModel

video_model = VideoClassCapsNetModel()


def videos_classification(npzfile):
    result = dict()
    i = 0
    for key in npzfile.files:
        result[i] = video_model.predict_short(npzfile[key])
        i += 1

    return result
