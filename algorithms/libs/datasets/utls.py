import shutil
from PIL import Image
import cv2
import numpy as np
import os
import math
import sys
import ssl
import hashlib

ssl._create_default_https_context = ssl._create_unverified_context


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def video_capturing_function(video_directory, dataset, folder_name,
                             image_width=250, image_height=250, image_gray=True, labels=None):
    for i in np.arange(len(dataset)):
        video_name = dataset.video_path[i]
        video_read_path = os.path.join(video_name)
        cap = cv2.VideoCapture(video_read_path)
        try:
            if labels is None:
                train_write_file = video_directory + os.path.sep + folder_name
            else:
                train_write_file = video_directory + os.path.sep + folder_name + os.path.sep + \
                                   labels[dataset.labels[i]]
            try:
                os.mkdir(train_write_file)
            except:
                pass
            train_write_file = os.path.join(train_write_file, hashlib.md5(os.path.basename(dataset.video_path[i]).split(".")[0].encode("utf-8")).hexdigest())
            os.mkdir(train_write_file)

            cap.set(cv2.CAP_PROP_FPS, 30)
            frame_rate = 1
            count = 0
            while cap.isOpened():
                frame_id = cap.get(1)  # current frame number
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % math.floor(frame_rate) == 0:
                    filename = "frame_%d.jpg" % count
                    count += 1
                    if image_gray:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_frame = cv2.resize(frame, (image_width, image_height))
                    cv2.imwrite(os.path.join(train_write_file, filename), resized_frame)
            cap.release()
        except Exception as e:
            print(e)

    return print("All frames written in the: " + folder_name + " Folder")


def frame_generating_function(dataset, dir_path, frame_size=10):
    for i in np.arange(len(dataset.video_name)):
        vid_name = dataset.video_name[i]
        vid_path = os.path.join(dir_path, os.path.basename(vid_name.split(".")[0]))
        len_frame = len(os.listdir(vid_path))
        j = frame_size - len(os.listdir(vid_path))
        if j > 0:
            c = 0
            for k in np.arange(j):
                list_frames = os.listdir(vid_path)
                frame = os.path.join(vid_path, list_frames[c])
                count = k + len_frame
                new_frame = "frame_%d.jpg" % count
                shutil.copy2(frame, os.path.join(vid_path, new_frame))
                c += 1
        else:
            pass
    return print("Frame Generation Done!")


def data_load_function_frames(dataset, directory, frame_size=10, image_width=250, image_height=250):
    frames = []
    for i in np.arange(len(dataset)):
        vid_name = dataset.video_name[i]
        vid_dir_path = os.path.join(directory, os.path.basename(vid_name.split(".")[0]))
        frames_to_select = []
        for j in np.arange(0, frame_size):
            frames_to_select.append('frame_%d.jpg' % j)
        vid_data = []
        for frame in frames_to_select:
            img = Image.open(os.path.join(vid_dir_path, frame))
            img = img.resize((image_width, image_height), Image.ANTIALIAS)
            data = np.asarray(img)
            norma_data = data / 255
            vid_data.append(norma_data)
        vid_data = np.array(vid_data)
        frames.append(vid_data)
    return np.array(frames)
