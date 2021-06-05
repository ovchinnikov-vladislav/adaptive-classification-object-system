import os
import cv2
import math
from libs.detection.utils import ObjectDetectionModel
import config


def prepare_event_frames(video_dir, frame_dir, image_width=250, image_height=250, image_gray=False):
    video_read_path = video_dir
    cap = cv2.VideoCapture(video_read_path)
    model = ObjectDetectionModel(classes=[class_name.split('\n')[0] for class_name in open(config.coco_classes_en).readlines()],
                                 weights='D:/MasterDissertation/models/yolov3.tf', model='yolo3', use_tracking=False)
    try:
        train_write_file = os.path.join(frame_dir, os.path.basename(video_dir).split('_rgb.')[0])
        os.makedirs(train_write_file)

        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_rate = 10
        count = 0
        while cap.isOpened():
            frame_id = cap.get(1)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                filename = "frame_%d.jpg" % count
                count += 1
                if image_gray:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, detections = model.detect_image(frame)
                for detection in detections:
                    if detection.get_class() == 'person':
                        x1, y1, x2, y2 = detection.get_box()
                        frame = frame[y1:y2, x1:x2]
                        resized_frame = cv2.resize(frame, (image_width, image_height))
                        cv2.imwrite(os.path.join(train_write_file, filename), resized_frame)
        cap.release()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    classes = ['A001', 'A002', 'A008', 'A009', 'A027', 'A028', 'A029', 'A030', 'A043']

    training = 70

    dataset_dir = 'D:/Downloads/nturgbd_rgb'
    frames_name_dir = 'nturgbd_rgb_frames'

    num_video = 0
    for name_dir in os.listdir(dataset_dir):
        try:
            index = classes.index(str(name_dir))
        except ValueError as e:
            index = -1

        if index == -1:
            continue

        videos = os.listdir(os.path.join(dataset_dir, name_dir))

        print(name_dir, index)

        count = 0
        for video in videos:
            skeleton_file = os.path.join(dataset_dir, 'skeleton', video.split('_rgb.')[0] + '.skeleton')
            prepare_event_frames(os.path.join(dataset_dir, name_dir, video),
                                 os.path.join(dataset_dir,  frames_name_dir))

            frame_path = frames_name_dir + '/' + video.split('_rgb.')[0]
            if count < len(videos) * 70 // 100:
                count += 1
                with open(os.path.join(dataset_dir, 'annotation-train.txt'), 'a') as annotation_train_file:
                    video_path = frame_path
                    annotation_train_file.write(video_path + ' ' + str(index) + '\n')
            else:
                with open(os.path.join(dataset_dir, 'annotation-test.txt'), 'a') as annotation_test_file:
                    video_path = frame_path
                    annotation_test_file.write(video_path + ' ' + str(index) + '\n')
            print(str(num_video + 1) + '.', video_path, 'class: ', str(index))
            num_video += 1
