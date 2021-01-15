import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
import numpy as np
import cv2
import random
import json


def drawmap(img, x, y, rot, mirror) :
    if mirror :
        tag = -1
    else:
        tag = 1
    color = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    cos = np.cos(-1.570796327 - rot)
    sin = np.sin(-1.570796327 - rot)
    camlines = [[], [], []]
    diff = [0., 0., 0.]
    for lane in lines:
        if not (lane[2] > x + 64 or lane[3] < x - 64 or lane[4] > y + 64 or lane[5] < y - 64) :
            line = np.array([ [((pt[0] - x) * cos - (pt[1] - y) * sin) * -20 * tag + 128, ((pt[0] - x) * sin + (pt[1] - y) * cos) * 20 + 500] for pt in lane[1] if -64 < (pt[0] - x) < 64 and -64 < (pt[1] - y) < 64 ], np.int32)
            if len(line) >= 2:
                cv2.polylines(img, [line], False, color[lane[0]], 2)

    

def drawonehot(img, x, y, rot, mirror) :
    if mirror :
        tag = -1
    else:
        tag = 1
    color = [1, 2, 3]
    cos = np.cos(-1.570796327 - rot)
    sin = np.sin(-1.570796327 - rot)
    camlines = [[], [], []]
    diff = [0., 0., 0.]
    for lane in lines:
        if not (lane[2] > x + 64 or lane[3] < x - 64 or lane[4] > y + 64 or lane[5] < y - 64) :
            line = np.array([ [((pt[0] - x) * cos - (pt[1] - y) * sin) * -20 * tag + 128, ((pt[0] - x) * sin + (pt[1] - y) * cos) * 20 + 500] for pt in lane[1] if -64 < (pt[0] - x) < 64 and -64 < (pt[1] - y) < 64 ], np.int32)
            if len(line) >= 2:
                cv2.polylines(img, [line], False, color[lane[0]], 2)

def network(input_cam):
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
input_map = tf.compat.v1.placeholder(tf.int32, [None, 256, 256])
input_gt = tf.compat.v1.one_hot(input_map, 4, axis=-1)



output = network(input_cam)
output_softmax = tf.compat.v1.nn.softmax(output)

cost = tf.compat.v1.losses.softmax_cross_entropy(input_gt, output) 
global_step = tf.compat.v1.placeholder(tf.int64)
learning_rate = tf.compat.v1.train.exponential_decay(0.001, global_step, 10, 0.9) 
operation = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)


lines = []
#read db
with open("map/kcity_final.mapdb") as f :
    js = json.load(f)
    for edge in js["Lines"]:
        line = np.float32(edge["line"])
        minx = np.min(line[:, 0])
        miny = np.min(line[:, 1])
        maxx = np.max(line[:, 0])
        maxy = np.max(line[:, 1])
        if edge["type"] // 100 == 2:
            lines.append([1, line, minx, maxx, miny, maxy])
        elif edge["type"] // 100 == 1 or edge["type"] // 100 == 3:
            lines.append([2, line, minx, maxx, miny, maxy])
    for edge in js["Stops"]:
        line = np.float32(edge["line"])
        minx = np.min(line[:, 0])
        miny = np.min(line[:, 1])
        maxx = np.max(line[:, 0])
        maxy = np.max(line[:, 1])
        lines.append([0, line, minx, maxx, miny, maxy])

image_path_list = []
image_path_list.extend(["/media/user/disk1/20201223_Simulation_Result/noise_no_y_0.3_0.06/" + x[:-8] for x in os.listdir("/media/user/disk1/20201223_Simulation_Result/noise_no_y_0.3_0.06/") if x[-7:] == "cam.png"])
image_path_list.extend(["/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1/" + x[:-8] for x in os.listdir("/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1/") if x[-7:] == "cam.png"])

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.Saver(max_to_keep=0)
#saver.restore(sess, "./log_wp_3/model_wp_1000.ckpt")
log_file = open("log_wp_1/log_wp_1.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


for epoch in range(1, 5001):
    for step in range(32) :
        dic = random.sample(image_path_list, k=8)
        seed = random.randint(0, 65535)
        cam_image = [cv2.imread(x + "_cam.png").astype(np.float32) / 255. for x in dic]

        map_image = np.zeros((8, 256, 256), dtype=np.int32)
        map_image_sample = np.zeros((256, 256, 3), dtype=np.float32)
        mapcpr_image = []
        for r in range(8):
            if random.random() > 0.5 :
                mirror = True
            else:
                mirror = False
            if mirror :
                cam_image[r] = cv2.flip(cam_image[r], 1)
            with open(dic[r] + "_arg.txt") as f :
                js = json.load(f)
                lat = float(js["x"]) + float(js["noise_x"])
                lon = float(js["y"]) + float(js["noise_y"])
                rot = float(js["yaw"]) + float(js["noise_yaw"])
                gt = [lat, lon, rot]

            drawonehot(map_image[r], gt[0], gt[1], gt[2], mirror)
            if r == 0:
                drawmap(map_image_sample, gt[0], gt[1], gt[2], mirror)

        _, ret, ret2 = sess.run((operation, output_softmax, cost), {input_cam:cam_image, input_map:map_image, global_step:epoch})
        cv2.imshow("cam_image", cam_image[0])
        cv2.imshow("map_image", map_image_sample)
        cv2.imshow("ans_image", ret[0, :, :, 1:])
        print("cost", ret2)
        cv2.waitKey(10)

        log_file.write("epoch " + str(epoch) + " step " + str(step) + " : " + str(ret2) + "\n")

    if epoch % 500 == 0:
        saver.save(sess, "log_wp_1/model_wp_" + str(epoch) + ".ckpt")
