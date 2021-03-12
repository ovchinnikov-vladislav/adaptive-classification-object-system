import cv2
import numpy as np
from PIL import Image
from urllib import request
from timeit import default_timer as timer
from bmstu.yolo3.model import YoloModel


face_cascade = cv2.CascadeClassifier('face-localization.xml')

def face_localization(yolo, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    r_image = yolo.detect_image(im_pil)
    numpy_image = np.array(r_image)
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def detect_video_webcam(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_video_ipcam(yolo, video_path):
    stream = request.urlopen(video_path)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = Image.fromarray(i)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

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
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow(video_path, i)
            if cv2.waitKey(1) == 27:
                return


if __name__ == '__main__':
    yolo = YoloModel()
    # vc = cv2.VideoCapture(0)
    # frame = None
    # if vc.isOpened():
    #     rval, frame = vc.read()
    #     frame = face_localization(yolo, frame)
    # else:
    #     rval = False
    #
    # accum_time = 0
    # curr_fps = 0
    # fps = "FPS: ??"
    # prev_time = timer()
    # while rval and frame is not None:
    #     curr_time = timer()
    #     exec_time = curr_time - prev_time
    #     prev_time = curr_time
    #     accum_time = accum_time + exec_time
    #     curr_fps = curr_fps + 1
    #     if accum_time > 1:
    #         accum_time = accum_time - 1
    #         fps = "FPS: " + str(curr_fps)
    #         curr_fps = 0
    #     cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=0.50, color=(255, 0, 0), thickness=2)
    #     cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #
    #     cv2.imshow("preview", frame)
    #
    #     rval, frame = vc.read()
    #     frame = face_localization(yolo, frame)
    #     key = cv2.waitKey(20)
    #     if key == 27:
    #         break
    #
    # vc.release()
    #detect_video_ipcam(yolo, 'http://192.168.0.16:8080/video')
    detect_video_ipcam(yolo, 'http://192.168.0.16:8080/video')

