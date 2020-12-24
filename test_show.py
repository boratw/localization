import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json


def getaffinemaxrix(ret):
    cos = np.cos(ret[2])
    sin = np.sin(ret[2])
    return np.array([[cos, sin, ret[0] * 12 - 736 * sin - 512 * cos + 512],
         [-sin, cos, ret[1] * 12 - 736 * cos + 512 * sin + 736]])

def network_uw(input_cam):
    with tf.compat.v1.variable_scope('Unwarp'):

        cam_conv0 = tf.compat.v1.nn.avg_pool(input_cam, [1, 2], [1, 2], "SAME") # 256 256
        cam_conv1 = tf.compat.v1.layers.conv2d(cam_conv0, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 128 128
        cam_conv1 = tf.compat.v1.nn.leaky_relu(cam_conv1, alpha=0.1)
        cam_conv2 = tf.compat.v1.layers.conv2d(cam_conv1, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 64 64
        cam_conv2 = tf.compat.v1.nn.leaky_relu(cam_conv2, alpha=0.1)
        cam_conv3 = tf.compat.v1.layers.conv2d(cam_conv2, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 32 32
        cam_conv3 = tf.compat.v1.nn.leaky_relu(cam_conv3, alpha=0.1)
        cam_conv4 = tf.compat.v1.layers.conv2d(cam_conv3, 512, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 16 16
        cam_conv4 = tf.compat.v1.nn.leaky_relu(cam_conv4, alpha=0.1)
        cam_conv5 = tf.compat.v1.layers.conv2d(cam_conv4, 1024, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 8 8 
        cam_conv5 = tf.compat.v1.nn.leaky_relu(cam_conv5, alpha=0.1)
        cam_conv6 = tf.compat.v1.layers.conv2d(cam_conv5, 2048, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 4 4 
        cam_conv6 = tf.compat.v1.nn.leaky_relu(cam_conv6, alpha=0.1)

        interm = tf.compat.v1.layers.conv2d(cam_conv6, 2048, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 4 4 

        map_conv5 = tf.compat.v1.layers.conv2d_transpose(interm, 2048, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True)  / 25 # 8 8
        map_conv5_2 = tf.compat.v1.layers.conv2d(map_conv5, 2048, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9
        map_conv5_2 = tf.compat.v1.nn.leaky_relu(map_conv5_2, alpha=0.1)
        map_conv4 = tf.compat.v1.layers.conv2d_transpose(map_conv5_2, 1024, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 16 16
        map_conv4_2 = tf.compat.v1.layers.conv2d(map_conv4, 512, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9
        map_conv4_2 = tf.compat.v1.nn.leaky_relu(map_conv4_2, alpha=0.1)
        map_conv3 = tf.compat.v1.layers.conv2d_transpose(map_conv4_2, 512, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 32 32
        map_conv3_2 = tf.compat.v1.layers.conv2d(map_conv3, 256, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 
        map_conv3_2 = tf.compat.v1.nn.leaky_relu(map_conv3_2, alpha=0.1)
        map_conv2 = tf.compat.v1.layers.conv2d_transpose(map_conv3_2, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 64 64
        map_conv2_2 = tf.compat.v1.layers.conv2d(map_conv2, 128, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9
        map_conv2_2 = tf.compat.v1.nn.leaky_relu(map_conv2_2, alpha=0.1)
        map_conv1 = tf.compat.v1.layers.conv2d_transpose(map_conv2_2, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 128 128
        map_conv1_2 = tf.compat.v1.layers.conv2d(map_conv1, 64, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9  
        map_conv1_2 = tf.compat.v1.nn.leaky_relu(map_conv1_2, alpha=0.1)
        map_conv0 = tf.compat.v1.layers.conv2d_transpose(map_conv1_2, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 256 256
        map_conv0_2 = tf.compat.v1.layers.conv2d(map_conv0, 4, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9

        return map_conv0_2

tf.compat.v1.disable_eager_execution()
input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])

output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
restorer.restore(sess, "./log_wp_3/model_wp_2000.ckpt")

directory = "/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1/"

for epoch in range(10, 1001):
    for step in range(128) :
        cam_image = cv2.imread(directory + "%06d_%03d_cam.png" % (epoch, step))
        if cam_image is not None:
            cam_image = cam_image.astype(np.float32) / 255.
            map_image = cv2.imread(directory + "%06d_%03d_map.png" % (epoch, step)).astype(np.float32) / 255.
            cv2.imshow("cam_image", cam_image)
            cv2.circle(map_image, (512,736), 2, (0,0,255), -1)
            cv2.imshow("map_image", map_image)
            cv2.imshow("mapcam_image", map_image[384:640, 384:640])


            M = getaffinemaxrix([0, 0, 0.1])
            globalmap_warped_image = cv2.warpAffine(map_image, M, (1024, 1024))
            cv2.circle(globalmap_warped_image, (512,736), 2, (0,0,255), -1)
            cv2.imshow("warp_image_0", globalmap_warped_image)

            M = getaffinemaxrix([0.5, 0, 0.1])
            globalmap_warped_image = cv2.warpAffine(map_image, M, (1024, 1024))
            cv2.circle(globalmap_warped_image, (512,736), 2, (0,0,255), -1)
            cv2.imshow("warp_image_1", globalmap_warped_image)

            M = getaffinemaxrix([0, 0.5, 0.1])
            globalmap_warped_image = cv2.warpAffine(map_image, M, (1024, 1024))
            cv2.circle(globalmap_warped_image, (512,736), 2, (0,0,255), -1)
            cv2.imshow("warp_image_2", globalmap_warped_image)

            ret_output_unwarp_softmax = sess.run(output_unwarp_softmax, {input_cam:[cam_image]})
            warpedcam_list = ret_output_unwarp_softmax[:, :, :, 1:]
            cv2.imshow("unwarp_image", warpedcam_list[0])

            cv2.waitKey(0)
        else:
            break
