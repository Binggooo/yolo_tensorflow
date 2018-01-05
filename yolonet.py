import tensorflow as tf


class YoloNet:

    INPUT_WIDTH = 448
    INPUT_HEIGHT = 448
    INPUT_CHANNELS = 3

    ALPHA_LEAKY_RELU = 0.1

    inputImage = None

    def __init__(self):
        self.inputImage = tf.placeholder('float32', [None, self.INPUT_WIDTH, self.INPUT_HEIGHT, self.INPUT_CHANNELS])  # Size: 448*448*3

        self._conv_1 = self._conv_layer(self.inputImage, 'Variable', 'Variable_1', (7, 7, 3, 64), 2)  # Size: 224*224*64
        self._pool_2 = self._max_pool(self._conv_1, 2, 2)  # Size: 112*112*64

        self._conv_3 = self._conv_layer(self._pool_2, 'Variable_2', 'Variable_3', (3, 3, 64, 192), 1)  # Size: 112*112*192
        self._pool_4 = self._max_pool(self._conv_3, 2, 2)  # Size: 56*56*192

        self._conv_5 = self._conv_layer(self._pool_4, 'Variable_4', 'Variable_5', (1, 1, 192, 128), 1)  # Size: 56*56*128
        self._conv_6 = self._conv_layer(self._conv_5, 'Variable_6', 'Variable_7', (3, 3, 128, 256), 1)  # Size: 56*56*256
        self._conv_7 = self._conv_layer(self._conv_6, 'Variable_8', 'Variable_9', (1, 1, 256, 256), 1)  # Size: 56*56*256
        self._conv_8 = self._conv_layer(self._conv_7, 'Variable_10', 'Variable_11', (3, 3, 256, 512), 1)  # Size: 56*56*512
        self._pool_9 = self._max_pool(self._conv_8, 2, 2)  # Size: 28*28*512

        self._conv_10 = self._conv_layer(self._pool_9, 'Variable_12', 'Variable_13', (1, 1, 512, 256), 1)  # Size: 28*28*256
        self._conv_11 = self._conv_layer(self._conv_10, 'Variable_14', 'Variable_15', (3, 3, 256, 512), 1)  # Size: 28*28*512
        self._conv_12 = self._conv_layer(self._conv_11, 'Variable_16', 'Variable_17', (1, 1, 512, 256), 1)  # Size: 28*28*256
        self._conv_13 = self._conv_layer(self._conv_12, 'Variable_18', 'Variable_19', (3, 3, 256, 512), 1)  # Size: 28*28*512
        self._conv_14 = self._conv_layer(self._conv_13, 'Variable_20', 'Variable_21', (1, 1, 512, 256), 1)  # Size: 28*28*256
        self._conv_15 = self._conv_layer(self._conv_14, 'Variable_22', 'Variable_23', (3, 3, 256, 512), 1)  # Size: 28*28*512
        self._conv_16 = self._conv_layer(self._conv_15, 'Variable_24', 'Variable_25', (1, 1, 512, 256), 1)  # Size: 28*28*256
        self._conv_17 = self._conv_layer(self._conv_16, 'Variable_26', 'Variable_27', (3, 3, 256, 512), 1)  # Size: 28*28*512
        self._conv_18 = self._conv_layer(self._conv_17, 'Variable_28', 'Variable_29', (1, 1, 512, 512), 1)  # Size: 28*28*512
        self._conv_19 = self._conv_layer(self._conv_18, 'Variable_30', 'Variable_31', (3, 3, 512, 1024), 1)  # Size: 28*28*1024
        self._pool_20 = self._max_pool(self._conv_19, 2, 2)  # Size: 14*14*1024

        self._conv_21 = self._conv_layer(self._pool_20, 'Variable_32', 'Variable_33', (1, 1, 1024, 512), 1)  # Size: 14*14*512
        self._conv_22 = self._conv_layer(self._conv_21, 'Variable_34', 'Variable_35', (3, 3, 512, 1024), 1)  # Size: 14*14*1024
        self._conv_23 = self._conv_layer(self._conv_22, 'Variable_36', 'Variable_37', (1, 1, 1024, 512), 1)  # Size: 14*14*512
        self._conv_24 = self._conv_layer(self._conv_23, 'Variable_38', 'Variable_39', (3, 3, 512, 1024), 1)  # Size: 14*14*1024
        self._conv_25 = self._conv_layer(self._conv_24, 'Variable_40', 'Variable_41', (3, 3, 1024, 1024), 1)  # Size: 14*14*1024
        self._conv_26 = self._conv_layer(self._conv_25, 'Variable_42', 'Variable_43', (3, 3, 1024, 1024), 2)  # Size: 7*7*1024
        self._conv_27 = self._conv_layer(self._conv_26, 'Variable_44', 'Variable_45', (3, 3, 1024, 1024), 1)  # Size: 7*7*1024
        self._conv_28 = self._conv_layer(self._conv_27, 'Variable_46', 'Variable_47', (3, 3, 1024, 1024), 1)  # Size: 7*7*1024

        self._fc_29 = self._fc_layer(self._conv_28, 'Variable_48', 'Variable_49', (50176, 512), isFlat=True, isRelu=True)  # Size: 512
        self._fc_30 = self._fc_layer(self._fc_29, 'Variable_50', 'Variable_51', (512, 4096), isFlat=False, isRelu=True)  # Size: 4096
        self._fc_31 = self._fc_layer(self._fc_30, 'Variable_52', 'Variable_53', (4096, 1470), isFlat=False, isRelu=False)  # Size: 1470

        print('YoloNet is built...')

    def _conv_layer(self, inputs, filter_name, bias_name, filter_shape, stride):
        # filter = tf.Variable(tf.truncated_normal(filter_shape, mean=0, stddev=0.1), dtype='float32')
        # bias = tf.Variable(tf.truncated_normal((filter_shape[3],), mean=0, stddev=0.1), dtype='float32')

        filter = tf.get_variable(filter_name, shape=filter_shape, dtype='float32')
        bias = tf.get_variable(bias_name, shape=(filter_shape[3]), dtype='float32')

        conv = tf.nn.conv2d(inputs, filter, strides=[1, stride, stride, 1], padding='SAME')
        conv_bias = tf.nn.bias_add(conv, bias)
        conv_relu = tf.nn.leaky_relu(conv_bias, self.ALPHA_LEAKY_RELU)
        return conv_relu

    def _max_pool(self, inputs, size, stride):
        return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

    def _fc_layer(self, inputs, theta_name, bias_name, theta_shape, isFlat=False, isRelu=True):
        # theta = tf.Variable(tf.truncated_normal(theta_shape, mean=0, stddev=0.1), dtype='float32')
        # bias = tf.Variable(tf.truncated_normal((theta_shape[1],), mean=0, stddev=0.1), dtype='float32')

        theta = tf.get_variable(theta_name, shape=theta_shape, dtype='float32')
        bias = tf.get_variable(bias_name, shape=(theta_shape[1]), dtype='float32')
        if isFlat is True:
            shape = inputs.get_shape()
            inputs = tf.reshape(tf.transpose(inputs, (0, 3, 1, 2)), (-1, shape[1] * shape[2] * shape[3]))

        fc = tf.nn.bias_add(tf.matmul(inputs, theta), bias)
        if isRelu is True:
            fc = tf.nn.leaky_relu(fc, self.ALPHA_LEAKY_RELU)
        return fc

    @property
    def output(self):
        classes = tf.reshape(self._fc_31[:, 0:980], (-1, 7, 7, 20))
        scales = tf.reshape(self._fc_31[:, 980:1078], (-1, 7, 7, 2))
        boxes = tf.reshape(self._fc_31[:, 1078:1470], (-1, 7, 7, 2, 4))
        print('prediction is OK...')
        return classes[0], scales[0], boxes[0]
