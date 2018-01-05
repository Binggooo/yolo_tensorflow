import numpy as np

THRESHOLD_CONFIDENCE = 0.2
THRESHOLD_IOU = 0.5


def _confidenceFilter(classes, scales, boxes):
    nh = scales.shape[0]
    nw = scales.shape[1]
    ns = scales.shape[2]
    nc = classes.shape[2]

    confidences = np.zeros((nh, nw, ns, nc))
    for i in range(ns):
        for j in range(nc):
            confidences[:, :, i, j] = np.multiply(classes[:, :, j], scales[:, :, i])

    filter = np.array(confidences > THRESHOLD_CONFIDENCE, dtype='bool')
    survivor = np.nonzero(filter)
    confidences_filtered = confidences[filter]
    boxes_filtered = boxes[survivor[0], survivor[1], survivor[2]]
    classes_filtered = np.argmax(filter, axis=3)[survivor[0], survivor[1], survivor[2]]

    sort = np.array(np.argsort(confidences_filtered))[::-1]
    confidences_sorted = confidences_filtered[sort]
    boxes_sorted = boxes_filtered[sort]
    classes_sorted = classes_filtered[sort]

    print('confidenceFilter is OK...')
    return confidences_sorted, boxes_sorted, classes_sorted


def _nmsFilter(confidences, boxes, classes):
    length = len(confidences)
    for i in range(length - 1):
        if confidences[i] == 0:
            continue
        for j in range(i + 1, length):
            if _iou(boxes[i, :], boxes[j, :]) > THRESHOLD_IOU:
                confidences[j] = 0

    filter = np.array(confidences > 0, dtype='bool')
    confidences_filtered = confidences[filter]
    boxes_filtered = boxes[filter]
    classes_filtered = classes[filter]

    print('iouFilter is OK...')
    return confidences_filtered, boxes_filtered, classes_filtered


def _iou(box1, box2):
    tb = min(box1[2], box2[2]) - max(box1[0], box2[0])
    lr = min(box1[3], box2[3]) - max(box1[1], box2[1])

    intersection = 0
    if tb > 0 and lr > 0:
        intersection = tb * lr
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
    return intersection / union


def resultFilter(classes, scales, boxes):
    confidences_sorted, boxes_sorted, classes_sorted = _confidenceFilter(classes, scales, boxes)
    confidences_filtered, boxes_filtered, classes_filtered = _nmsFilter(confidences_sorted, boxes_sorted, classes_sorted)
    return confidences_filtered, boxes_filtered, classes_filtered
