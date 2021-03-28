import cv2
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from libs.yolo3.utils import transform_images, convert_boxes
from libs.deepsort import preprocessing, nn_matching
from libs.deepsort.detection import Detection
from libs.deepsort.tracker import Tracker
from libs.deepsort.box_encoder import create_box_encoder
from libs.yolo3.layers import yolo_v3


if __name__ == '__main__':
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/deepsort.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    yolo = yolo_v3()

    yolo.load_weights('./model_data/yolov3.tf')

    class_names = [c.strip() for c in open('./model_data/coco_classes.txt').readlines()]

    vid = cv2.VideoCapture('http://192.168.0.16:8080/video')

    fps = 0.0
    count = 0
    while True:
        _, img = vid.read()

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature)
                      for bbox, score, class_name, feature
                      in zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                           int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id),
                        (int(bbox[0]), int(bbox[1] - 10)),
                        0, 0.75, (255, 255, 255), 2)

        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        cv2.imshow('output', img)

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
