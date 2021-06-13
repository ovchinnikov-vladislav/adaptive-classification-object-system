from libs.capsnets.utils import VideoClassCapsNetModel

video_model = VideoClassCapsNetModel()


def classification_video(video):
    return video_model.predict_short(video)
