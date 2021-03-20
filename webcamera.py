import cv2
import streamlink
import numpy as np
from PIL import Image
from urllib import request
from timeit import default_timer as timer
from bmstu.yolo3.model import YoloModel
import time


def image_detection(yolo, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_image, object_detection = yolo.detect_image(img)
    numpy_image = np.array(r_image)
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)


def detect_video_webcam(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    # if not vid.isOpened():
    #     raise IOError("Couldn't open webcam or video")
    # video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    # video_fps = vid.get(cv2.CAP_PROP_FPS)
    # video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #               int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # isOutput = True if output_path != "" else False
    # if isOutput:
    #     print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    #     out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    # accum_time = 0
    # curr_fps = 0
    # fps = "FPS: ??"
    # prev_time = timer()
    while True:
        return_value, frame = vid.read()
        # image = Image.fromarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = yolo.detect_image(frame)
        # result = np.asarray(image)
        # curr_time = timer()
        # exec_time = curr_time - prev_time
        # prev_time = curr_time
        # accum_time = accum_time + exec_time
        # curr_fps = curr_fps + 1
        # if accum_time > 1:
        #     accum_time = accum_time - 1
        #     fps = "FPS: " + str(curr_fps)
        #     curr_fps = 0
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", image)
        # if isOutput:
        #     out.write(result)
        if cv2.waitKey(1) == 27:
            break


def detect_video_ipcam(yolo, video_path):
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    # bytes = b''
    while True:
        im = Image.open(request.urlopen(video_path))
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        image = yolo.detect_image(im)
        # result = np.asarray(image)
        # curr_time = timer()
        # exec_time = curr_time - prev_time
        # prev_time = curr_time
        # accum_time = accum_time + exec_time
        # curr_fps = curr_fps + 1
        # if accum_time > 1:
        #     accum_time = accum_time - 1
        #     fps = "FPS: " + str(curr_fps)
        #     curr_fps = 0
        # cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.50, color=(255, 0, 0), thickness=2)
        # cv2.imshow("result", result)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", image)
        #  cv2.imwrite('shot.jpg', result)

        if cv2.waitKey(1) == 27:
            break


def detect_video_ipcam_print_image(yolo, video_path):
    fps = "FPS: ??"
    while True:
        im = Image.open(request.urlopen(video_path))
        im = np.array(im)

        t1 = time.time()
        image, result = yolo.detect_image(im)
        t2 = time.time()
        print(f'time: {t2 - t1}')
        # result = np.asarray(image)
        fps = "FPS: " + str(int(1000 // int(1000 * (t2 - t1))))
        cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", image)
        #  cv2.imwrite('shot.jpg', image)

        if cv2.waitKey(1) == 27:
            break


def get_video(video_path):
    stream = request.urlopen(video_path)

    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow(video_path, i)
            if cv2.waitKey(1) == 27:
                return


if __name__ == '__main__':
    import tensorflow as tf

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    #  video_id = 'ENvmK1x0ZTc'
    #  url = f'https://www.youtube.com/watch?v={video_id}'

    #  streams = streamlink.streams(url)
    #  print(streams)
    #
    yolo = YoloModel()
    # detect_video_webcam(yolo, streams["360p"].url)
    # detect_video_ipcam(yolo, 'http://192.168.0.16:8080/shot.jpg')
    # detect_video_webcam(yolo, 'rtsp://192.168.0.16:5554/out.h264')
    #  detect_video_webcam(yolo, 'rtsp://10.75.118.98:5554/out.h264')
    detect_video_ipcam_print_image(yolo, 'http://192.168.0.16:8080/shot.jpg')
    # img = Image.open('416x416.jpg')
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #
    # img, object_detection = yolo.detect_image(img)
