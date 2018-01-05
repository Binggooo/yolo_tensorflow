import yolonet
import imagetools
import resulttools
import tensorflow as tf

PATH_READ = './examples/test_1.jpg'
PATH_WRITE = './examples/output_1.jpg'

# build YoloNet
yolo = yolonet.YoloNet()

# load image
img, img_array = imagetools.imageLoad(PATH_READ, yolo.INPUT_HEIGHT, yolo.INPUT_WIDTH)
feed_dict = {yolo.inputImage: img_array}

with tf.Session() as sess:
    # load pre-training data
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './YOLO_small.ckpt')

    # predict and transform
    classes, scales, boxes = sess.run(yolo.output, feed_dict=feed_dict)
    boxes_transform = imagetools.xyTransform(boxes, img.shape[0], img.shape[1])

    # filter and draw
    confidences, boxes, classes = resulttools.resultFilter(classes, scales, boxes_transform)
    imagetools.imageDraw(img, boxes, classes, confidences, PATH_WRITE, True)
