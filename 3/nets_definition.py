from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc

from layers_slim import *


def FCN_Seg(self, is_training=True):

    # Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {"is_training": self.is_training}

    print("input", self.tgt_image)

    with tf.variable_scope("First_conv"):
        conv1 = tc.layers.conv2d(
            self.tgt_image,
            32,
            3,
            1,
            normalizer_fn=self.normalizer,
            normalizer_params=self.bn_params,
        )

        print("Conv1 shape")
        print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0, self.normalizer, self.bn_params, 1)
    # print("Conv 1")
    # print(x.get_shape())

    # 180x180x24
    x = inverted_bottleneck(x, 6, 24, 1, self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0, self.normalizer, self.bn_params, 3)

    print("Block One dim ")
    print(x)

    DB2_skip_connection = x
    # 90x90x32
    x = inverted_bottleneck(x, 6, 32, 1, self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0, self.normalizer, self.bn_params, 5)

    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    # 45x45x96
    x = inverted_bottleneck(x, 6, 64, 1, self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0, self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0, self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0, self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0, self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0, self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0, self.normalizer, self.bn_params, 12)

    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    # 23x23x160
    x = inverted_bottleneck(x, 6, 160, 1, self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0, self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0, self.normalizer, self.bn_params, 15)

    print("Block Four dim ")
    print(x)

    # 23x23x320
    x = inverted_bottleneck(x, 6, 320, 0, self.normalizer, self.bn_params, 16)

    print("Block Four dim ")
    print(x)

    def upsample(tensor, match, factor, feature_maps, config, kernel_size=3):
        upsampled = slim.conv2d_transpose(
            tensor,
            feature_maps,
            kernel_size,
            factor,
            scope="config_{}_up_{}_maps_{}".format(config, factor, feature_maps),
            activation_fn=tf.nn.elu,
        )

        # h/w ratio is 1:1
        if shape(upsampled)[1] > shape(match)[1]:
            return {"upsampled": crop(upsampled, match), "match": match}
        else:
            return {"upsampled": upsampled, "match": crop(match, upsampled)}

    def refinement_block(tensor, skip_connection, factor, feature_maps, config, conv):
        res = upsample(tensor, skip_connection, factor, feature_maps, config)
        res = tf.concat([res["upsampled"], res["match"]], 3)
        res = tc.layers.conv2d(
            res,
            feature_maps,
            3,
            1,
            activation_fn=tf.nn.elu,
            normalizer_fn=self.normalizer,
            normalizer_params=self.bn_params,
            scope="config_{}_conv_{}".format(config, conv)
        )
        return res

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        # input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x
        # and produces 120 output feature maps, which are 16x bigger in resolution than
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        current_up5 = upsample(x, self.tgt_image, 16, 120, 1, kernel_size=16)

        End_maps_decoder1 = slim.conv2d(
            current_up5["upsampled"], self.N_classes, [1, 1], scope="Final_decoder"
        )  # (batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        # input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up3 = refinement_block(x, DB4_skip_connection, 2, 256, 2, 1)

        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1)
        # and produces 120 output feature maps, which are 8x bigger in resolution than
        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        current_up3 = upsample(current_up3, self.tgt_image, 8, 120, 2, kernel_size=8)

        End_maps_decoder1 = slim.conv2d(
            current_up3["upsampled"], self.N_classes, [1, 1], scope="Final_decoder"
        )  # (batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        # input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up4 = refinement_block(x, DB4_skip_connection, 2, 256, 3, 1)

        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        current_up4 = refinement_block(current_up4, DB3_skip_connection, 2, 160, 3, 2)

        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)
        # and produces 120 output feature maps which are 4x bigger in resolution than
        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4

        current_up4 = upsample(current_up4, self.tgt_image, 4, 120, 3, kernel_size=4)

        End_maps_decoder1 = slim.conv2d(
            current_up4["upsampled"], self.N_classes, [1, 1], scope="Final_decoder"
        )  # (batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Full configuration
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################

        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps

        current_up5 = refinement_block(x, DB4_skip_connection, 2, 256, 4, 1)

        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.

        current_up5 = refinement_block(current_up5, DB3_skip_connection, 2, 160, 4, 2)

        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.

        current_up5 = refinement_block(current_up5, DB2_skip_connection, 2, 96, 4, 3)

        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3)
        # and produce 120 output feature maps which are 2x bigger in resolution than
        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4

        current_up5 = upsample(current_up5, self.tgt_image, 2, 120, 4)

        End_maps_decoder1 = slim.conv2d(
            current_up5["upsampled"], self.N_classes, [1, 1], scope="Final_decoder"
        )  # (batchsize, width, height, N_classes)

        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    return Reshaped_map

