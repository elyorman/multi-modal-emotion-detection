from __future__ import print_function
import numpy as np
import cv2
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split

import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import numpy as np
import cv2
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Polygon
from sklearn.model_selection import train_test_split

import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor)


def global_pooll(input_tensor, pool_op=tf.nn.avg_pool):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size = tf.convert_to_tensor([1, tf.shape(input_tensor)[1], tf.shape(input_tensor)[2], 1])
    else:
        kernel_size = [1, shape[1], shape[2], 1]
    output = pool_op(
        input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                           padding='SAME')
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           rate=rate, padding='VALID', weights_regularizer=slim.l2_regularizer(0.0001),
                           weights_initializer=slim.variance_scaling_initializer())


def resBlock(x, Depth, depth_bottleneck, kernel_size=[3, 3], stride=1, skipCon=False, rate=2):
    depth_in = slim.utils.last_dimension(x.get_shape(), min_rank=4)
    peatct = slim.batch_norm(x, activation_fn=tf.nn.relu)
    if Depth == depth_in:
        shortcut = subsample(x, stride)
    else:
        shortcut = slim.conv2d(peatct, Depth, [1, 1], stride=stride,
                               activation_fn=None)
    residual = slim.conv2d(peatct, depth_bottleneck, [1, 1], stride=1, weights_regularizer=slim.l2_regularizer(0.0001),
                           weights_initializer=slim.variance_scaling_initializer())
    residual = tf.nn.relu(slim.batch_norm(residual, fused=True, scale=True))
    residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate)
    residual = tf.nn.relu(slim.batch_norm(residual, fused=True, scale=True))
    residual = slim.conv2d(residual, Depth, [1, 1], stride=1, activation_fn=None)

    output = shortcut + residual
    return output


def UnitBlockA(x, base_depth, stride=1, rate=1):
    """
    A custom Block: Bekmirzaev shohrukh
    """

    depth = base_depth * 4
    depth_bottleneck = base_depth
    res = resBlock(x, depth, depth_bottleneck, stride=1, skipCon=True, rate=rate)
    res = resBlock(res, depth, depth_bottleneck, stride=1, skipCon=False, rate=rate)
    res = resBlock(res, depth, depth_bottleneck, stride=stride, skipCon=True, rate=rate)
    return res


def UnitBlockB(x, base_depth, stride=1, rate=1):
    """
    B custom Block: Bekmirzaev shohrukh

    """
    depth = base_depth * 4
    depth_bottleneck = base_depth
    res = resBlock(x, depth, depth_bottleneck, stride=1, skipCon=True, rate=rate)
    res = resBlock(res, depth, depth_bottleneck, stride=1, skipCon=False, rate=rate)
    res = resBlock(res, depth, depth_bottleneck, stride=1, skipCon=False, rate=rate)
    res = resBlock(res, depth, depth_bottleneck, stride=stride, skipCon=True, rate=rate)
    res = tf.nn.relu(res)
    return res


def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0, stride=1):
    out = slim.conv2d(inputs, n_filters, kernel_size, stride=stride, activation_fn=None, normalizer_fn=None)
    return out


def ourCustomNetwork(inputs, is_training=True, scope='OurCustomNetwork', num_classes=2):
    net = conv_block(inputs, 32, stride=2)
    net = UnitBlockA(net, 64, stride=2)
    net = UnitBlockB(net, 64, stride=2)
    net = UnitBlockA(net, 128, stride=2)
    net = UnitBlockB(net, 256, stride=2)

    with tf.variable_scope(scope):
        net = global_pooll(net)
        # 1 x 1 x num_classes
        # Note: legacy scope name.
        logits = slim.conv2d(
            net,
            num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            biases_initializer=tf.zeros_initializer(),
            scope='Conv2d_1c_1x1')
        logits = tf.squeeze(logits, [1, 2])
        logits = tf.identity(logits, name='output')
    return logits