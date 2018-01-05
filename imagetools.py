import cv2
import numpy as np

CLASSES_TEXT = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def imageLoad(path, height_format, width_format):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (height_format, width_format))
    img_array = np.asarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    img_format = img_array.reshape((1, height_format, width_format, 3)) / 255.0 * 2.0 - 1.0
    print('Imgae is loaded...')
    return img, img_format


def xyTransform(boxes, height, width):
    boxes_cp = boxes.copy()
    nh = boxes.shape[0]
    nw = boxes.shape[1]
    nc = boxes.shape[2]
    single_h = height / nh
    single_w = width / nw

    for i in range(nh):
        for j in range(nw):
            for k in range(nc):
                centerX = (j + boxes[i, j, k, 0]) * single_w
                centerY = (i + boxes[i, j, k, 1]) * single_h
                width_half = boxes[i, j, k, 2] * boxes[i, j, k, 2] * width / 2
                height_half = boxes[i, j, k, 3] * boxes[i, j, k, 3] * height / 2
                boxes_cp[i, j, k, 0] = centerX - width_half
                boxes_cp[i, j, k, 1] = centerY - height_half
                boxes_cp[i, j, k, 2] = centerX + width_half
                boxes_cp[i, j, k, 3] = centerY + height_half
    print('xyTransform is OK...')
    return boxes_cp


def imageDraw(img, boxes, classes, confidences, path=None, isShow=False):
    img_cp = img.copy()
    boxes = boxes.astype(np.int32)
    for i in range(boxes.shape[0]):
        cv2.rectangle(img_cp, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), (0, 255, 0), 2)
        cv2.rectangle(img_cp, (boxes[i, 0], boxes[i, 1] - 20), (boxes[i, 2], boxes[i, 1]), (200, 200, 200), -1)
        cv2.putText(img_cp, CLASSES_TEXT[classes[i]] + ':%.2f' % confidences[i], (boxes[i, 0] + 5, boxes[i, 1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    print('imageDraw is OK...')

    if path is not None:
        cv2.imwrite(path, img_cp)
    if isShow is True:
        cv2.imshow('YOLO_small Detection', img_cp)
        cv2.waitKey(0)
