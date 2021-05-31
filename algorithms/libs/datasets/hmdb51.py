import requests
import os
import patoolib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import libs.datasets.utls as utils
import hashlib


def __download_hmdb51(data_dir_path, classes=[]):
    hmdb_rar = os.path.join(data_dir_path, 'hmdb51_org.rar')

    hmdb_link = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'

    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    if not os.path.exists(hmdb_rar):
        print('hmdb51 file does not exist, downloading from Internet')
        r = requests.get(hmdb_link)
        with open(hmdb_rar, 'wb') as outfile:
            outfile.write(r.content)

    print('unzipping hmdb51 file')
    patoolib.extract_archive(hmdb_rar, outdir=data_dir_path)
    os.remove(hmdb_rar)

    downloads_dir = os.listdir(data_dir_path)
    for arch_d in downloads_dir:
        if not len(classes) or arch_d.split('.')[0] in classes:
            patoolib.extract_archive(os.path.join(data_dir_path, arch_d), outdir=data_dir_path)
        os.remove(os.path.join(data_dir_path, arch_d))


def __scan_hmdb51(data_dir_path, limit):
    input_data_dir_path = os.path.join(data_dir_path)

    result = dict()

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        __help_scan_hmdb51(input_data_dir_path, f, dir_count, result)
        if dir_count == limit:
            break
    return result


def __help_scan_hmdb51(input_data_dir_path, f, dir_count, result):
    file_path = os.path.join(input_data_dir_path, f)
    if not os.path.isfile(file_path):
        dir_count += 1
        for ff in os.listdir(file_path):
            video_file_path = os.path.join(file_path, ff)
            result[video_file_path] = f


def __scan_hmdb51_with_labels(data_dir_path, labels):
    input_data_dir_path = os.path.join(data_dir_path)

    result = dict()

    dir_count = 0
    for label in labels:
        __help_scan_hmdb51(input_data_dir_path, label, dir_count, result)
    return result


def download_data(data_dist_path, image_width=250, image_height=250, image_gray=False, classes=[]):
    hmdb51_data_dir_path = os.path.join(data_dist_path)
    if not os.path.exists(hmdb51_data_dir_path):
        __download_hmdb51(data_dist_path, classes)

    videos = []
    labels = []
    name_class_labels = dict()

    dir_count = 0
    for f in os.listdir(hmdb51_data_dir_path):
        file_path = os.path.join(hmdb51_data_dir_path, f)
        print(file_path)
        if not os.path.isfile(file_path):
            dir_count += 1
            for video in os.listdir(file_path):
                videos.append(os.path.join(file_path, video))
                labels.append(dir_count - 1)
                name_class_labels[dir_count - 1] = f

    videos = pd.DataFrame(videos, labels).reset_index()
    videos.columns = ["labels", "video_path"]
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

    train_videos_dir = os.path.join(hmdb51_data_dir_path, "hmdb51_frames")
    test_videos_dir = os.path.join(hmdb51_data_dir_path, "hmdb51_frames")
    try:
        os.rmdir(train_videos_dir)
    except FileNotFoundError as e:
        print(train_videos_dir + " not exists, then create")
    os.mkdir(train_videos_dir)
    try:
        os.rmdir(test_videos_dir)
    except FileNotFoundError as e:
        print(test_videos_dir + " not exists, then create")
    os.mkdir(test_videos_dir)

    train_set = shuffle(train_set)
    with open(os.path.join(hmdb51_data_dir_path, 'annotation-train.txt'), 'w') as annotation_train_file:
        for i in np.arange(len(train_set)):
            video_name = hashlib.md5(os.path.basename(train_set.video_path[i]).split(".")[0].encode("utf-8")).hexdigest()
            video_path = os.path.abspath(os.path.join(hmdb51_data_dir_path, "hmdb51_frames", video_name))
            label = train_set.labels[i]
            annotation_train_file.write(video_path + ' ' + str(label) + '\n')

    test_set = shuffle(test_set)
    with open(os.path.join(hmdb51_data_dir_path, 'annotation-test.txt'), 'w') as annotation_test_file:
        for i in np.arange(len(test_set)):
            video_name = hashlib.md5(os.path.basename(test_set.video_path[i]).split(".")[0].encode("utf-8")).hexdigest()
            video_path = os.path.abspath(os.path.join(hmdb51_data_dir_path, "hmdb51_frames", video_name))
            label = test_set.labels[i]
            annotation_test_file.write(video_path + ' ' + str(label) + '\n')

    utils.video_capturing_function(hmdb51_data_dir_path, train_set, "hmdb51_frames", image_width, image_height, image_gray, None)
    utils.video_capturing_function(hmdb51_data_dir_path, test_set, "hmdb51_frames", image_width, image_height, image_gray, None)


if __name__ == '__main__':
    classes = ['chew', 'clap', 'drink', 'eat', 'fall_floor', 'jump', 'laugh', 'pick', 'run', 'sit', 'smile', 'stand', 'talk', 'walk']
    download_data(os.path.join('D:' + os.path.sep + 'tensorflow_datasets', 'hmdb51'), 160, 120, classes=classes)
