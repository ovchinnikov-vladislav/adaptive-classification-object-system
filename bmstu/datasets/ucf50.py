import urllib.request
import os
import patoolib
import pandas as pd
import numpy as np
import math
import bmstu.datasets.utls as utils
import bmstu.dynamic_images.utls as du
import cv2
import shutil


def __download_ucf50(data_dir_path):
    ucf_rar = data_dir_path + '/UCF50.rar'

    URL_LINK = 'https://www.crcv.ucf.edu/data/UCF50.rar'

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    if not os.path.exists(ucf_rar):
        print('ucf file does not exist, downloading from Internet')
        urllib.request.urlretrieve(url=URL_LINK, filename=ucf_rar,
                                   reporthook=utils.reporthook)

    print('unzipping ucf file')
    patoolib.extract_archive(ucf_rar, outdir=data_dir_path)


def __scan_ucf50(data_dir_path, limit):
    input_data_dir_path = data_dir_path + '/UCF50'

    result = dict()

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        __help_scan_ucf50(input_data_dir_path, f, dir_count, result)
        if dir_count == limit:
            break
    return result


def __help_scan_ucf50(input_data_dir_path, f, dir_count, result):
    file_path = input_data_dir_path + os.path.sep + f
    if not os.path.isfile(file_path):
        dir_count += 1
        for ff in os.listdir(file_path):
            video_file_path = file_path + os.path.sep + ff
            result[video_file_path] = f


def __scan_ucf50_with_labels(data_dir_path, labels):
    input_data_dir_path = data_dir_path + '/UCF50'

    result = dict()

    dir_count = 0
    for label in labels:
        __help_scan_ucf50(input_data_dir_path, label, dir_count, result)
    return result


def get_dynamic_images_data(data_dist_path):
    ucf50_data_dir_path = data_dist_path + "/UCF50"
    if not os.path.exists(ucf50_data_dir_path):
        __download_ucf50(data_dist_path)

    train_dir = os.path.join(ucf50_data_dir_path, 'train')
    test_dir = os.path.join(ucf50_data_dir_path, 'test')
    validation_dir = os.path.join(ucf50_data_dir_path, 'validation')

    try:
        try:
            shutil.rmtree(train_dir)
        except FileNotFoundError as e:
            print(train_dir + " not exists, then create")
        os.mkdir(train_dir)
    except FileExistsError as ae:
        print("Folder Already Created")

    try:
        try:
            shutil.rmtree(test_dir)
        except FileNotFoundError as e:
            print(test_dir + " not exists, then create")
        os.mkdir(test_dir)
    except FileExistsError as ae:
        print("Folder Already Created")

    try:
        try:
            shutil.rmtree(validation_dir)
        except FileNotFoundError as e:
            print(validation_dir + " not exists, then create")
        os.mkdir(validation_dir)
    except FileExistsError as ae:
        print("Folder Already Created")

    for f in os.listdir(ucf50_data_dir_path):
        file_path = ucf50_data_dir_path + os.path.sep + f
        if file_path == train_dir or file_path == test_dir or file_path == validation_dir:
            continue
        i = 0
        if not os.path.isfile(file_path):
            for video in os.listdir(file_path):
                frames = du.get_video_frames(file_path + os.path.sep + video)
                print(f'get frames by {file_path + os.path.sep + video}')
                dyn_image = du.get_dynamic_image(frames, normalized=True)
                cv2.imwrite(os.path.join(ucf50_data_dir_path, f + '_' + str(i) + '.jpg'), dyn_image)
                i += 1
        try:
            shutil.rmtree(file_path)
        except FileNotFoundError as e:
            print(file_path + ' not exists')
        except NotADirectoryError as e:
            print(file_path + ' not directory')

    images = [os.path.join(ucf50_data_dir_path, name) for name in os.listdir(ucf50_data_dir_path)
              if os.path.isfile(os.path.join(ucf50_data_dir_path, name))]

    class_images = dict()

    for elem in images:
        class_name = os.path.basename(elem).split('_')[0]
        if class_images.get(class_name) is None:
            class_images[class_name] = []
        class_images[class_name].append(elem)

    for key in class_images.keys():
        images = class_images[key]

        train_images_count = len(images) - len(images) // 3
        test_images_count = train_images_count - train_images_count // 2

        train_images = [images[i] for i in range(0, train_images_count)]
        for elem in train_images:
            basename = os.path.basename(elem)
            class_name = basename.split('_')[0]
            if not os.path.exists(os.path.join(train_dir, class_name)):
                os.mkdir(os.path.join(train_dir, class_name))
            shutil.move(elem, os.path.join(train_dir, class_name, basename))

        test_images = [images[i] for i in range(train_images_count, train_images_count + test_images_count // 2)]
        for elem in test_images:
            basename = os.path.basename(elem)
            class_name = basename.split('_')[0]
            if not os.path.exists(os.path.join(test_dir, class_name)):
                os.mkdir(os.path.join(test_dir, class_name))
            shutil.move(elem, os.path.join(test_dir, class_name, basename))

        validation_images = [images[i] for i in range(train_images_count + test_images_count // 2, len(images))]
        for elem in validation_images:
            basename = os.path.basename(elem)
            class_name = basename.split('_')[0]
            if not os.path.exists(os.path.join(validation_dir, class_name)):
                os.mkdir(os.path.join(validation_dir, class_name))
            shutil.move(elem, os.path.join(validation_dir, class_name, basename))

    return train_dir, test_dir, validation_dir


def load_data(data_dist_path, frame_size=10, image_width=250, image_height=250):
    ucf50_data_dir_path = data_dist_path + "/UCF50"
    if not os.path.exists(ucf50_data_dir_path):
        __download_ucf50(data_dist_path)

    videos = []
    labels = []
    name_class_labels = dict()

    dir_count = 0
    for f in os.listdir(ucf50_data_dir_path):
        file_path = ucf50_data_dir_path + os.path.sep + f
        print(file_path)
        if not os.path.isfile(file_path):
            dir_count += 1
            for video in os.listdir(file_path):
                videos.append(file_path + os.path.sep + video)
                labels.append(dir_count - 1)
                name_class_labels[dir_count - 1] = f

    videos = pd.DataFrame(videos, labels).reset_index()
    videos.columns = ["labels", "video_name"]
    videos.groupby('labels').count()

    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for i in set(labels):
        vs = videos.loc[videos["labels"] == i]
        vs_range = np.arange(len(vs))
        np.random.seed(12345)
        np.random.shuffle(vs_range)

        vs = vs.iloc[vs_range]
        last_train = len(vs) - len(vs) // 3
        train_vs = vs.iloc[:last_train]
        train_set = train_set.append(train_vs)
        test_vs = vs.iloc[last_train:]
        test_set = test_set.append(test_vs)

    train_set = train_set.reset_index().drop("index", axis=1)
    test_set = test_set.reset_index().drop("index", axis=1)

    train_videos_dir = os.path.join(ucf50_data_dir_path, "Train_Videos")
    test_videos_dir = os.path.join(ucf50_data_dir_path, "Test_Videos")
    try:
        try:
            os.rmdir(train_videos_dir)
        except FileNotFoundError as e:
            print(train_videos_dir + " not exists, then create")
        os.mkdir(train_videos_dir)
    except FileExistsError as ae:
        print("Folder Already Created")
    try:
        try:
            os.rmdir(test_videos_dir)
        except FileNotFoundError as e:
            print(test_videos_dir + " not exists, then create")
        os.mkdir(test_videos_dir)
    except FileExistsError as ae:
        print("Folder Already Created")

    utils.video_capturing_function(ucf50_data_dir_path, train_set, "Train_Videos")
    utils.video_capturing_function(ucf50_data_dir_path, test_set, "Test_Videos")

    train_dir_path = ucf50_data_dir_path + os.path.sep + 'Train_Videos'
    test_dir_path = ucf50_data_dir_path + os.path.sep + 'Test_Videos'

    train_frames = []
    for i in np.arange(len(train_set.video_name)):
        vid_file_name = os.path.basename(train_set.video_name[i]).split(".")[0]
        train_frames.append(len(os.listdir(os.path.join(train_dir_path, vid_file_name))))

    test_frames = []
    for i in np.arange(len(test_set.video_name)):
        vid_file_name = os.path.basename(test_set.video_name[i]).split('.')[0]
        test_frames.append(len(os.listdir(os.path.join(test_dir_path, vid_file_name))))

    utils.frame_generating_function(train_set, train_dir_path, frame_size=frame_size)
    utils.frame_generating_function(test_set, test_dir_path, frame_size=frame_size)

    train_vid_dat = pd.DataFrame()
    validation_vid_dat = pd.DataFrame()
    for label in set(labels):
        label_dat = train_set.loc[train_set["labels"] == label]
        train_len_label = math.floor(len(label_dat) * 0.80)

        train_dat_label = label_dat.iloc[:train_len_label]
        validation_dat_label = label_dat.iloc[train_len_label:]

        train_vid_dat = train_vid_dat.append(train_dat_label, ignore_index=True)
        validation_vid_dat = validation_vid_dat.append(validation_dat_label, ignore_index=True)

    train_dataset = utils.data_load_function_frames(train_vid_dat, train_dir_path,
                                                    frame_size=frame_size,
                                                    image_width=image_width,
                                                    image_height=image_height)
    test_dataset = utils.data_load_function_frames(test_set, test_dir_path,
                                                   frame_size=frame_size,
                                                   image_width=image_width,
                                                   image_height=image_height)
    validation_dataset = utils.data_load_function_frames(validation_vid_dat, train_dir_path,
                                                         frame_size=frame_size,
                                                         image_width=image_width,
                                                         image_height=image_height)

    train_labels = np.array(train_vid_dat.labels)
    test_labels = np.array(test_set.labels)
    validation_labels = np.array(validation_vid_dat.labels)

    return (train_dataset, train_labels), (test_dataset, test_labels), (validation_dataset, validation_labels)


if __name__ == '__main__':
    train_dir, test_dir, validation_dir = get_dynamic_images_data('/tmp')
