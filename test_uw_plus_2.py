import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json



def getaffinemaxrix(ret, forwardpath, backwardpath) :
    normx = 0.0
    normy = 0.0
    finalx = 0.0
    finaly = 0.0
    if ret[1] >= 0. :
        dist = ret[1]
        prevp = [0., 0.]
        for p in forwardpath :
            d = np.sqrt((p[0] - prevp[0]) ** 2 + (p[1] - prevp[1]) ** 2)
            if d > dist :
                normx = ((p[0] - prevp[0]) / d)
                normy = ((p[1] - prevp[1]) / d)
                finalx = prevp[0] + normx * dist
                finaly = prevp[1] + normy * dist
            elif p != forwardpath[-1] :
                prevp[0] = p[0]
                prevp[1] = p[1]
                dist -= d
            else :
                normx = ((p[0] - prevp[0]) / d)
                normy = ((p[1] - prevp[1]) / d)
                finalx = prevp[0] + normx * dist
                finaly = prevp[1] + normy * dist
    else:
        dist = -ret[1]
        prevp = [0., 0.]
        for p in backwardpath :
            d = np.sqrt((p[0] - prevp[0]) ** 2 + (p[1] - prevp[1]) ** 2)
            if d > dist :
                normx = ((prevp[0] - p[0]) / d)
                normy = ((prevp[1] - p[1]) / d)
                finalx = prevp[0] - normx * dist
                finaly = prevp[1] - normy * dist
            elif p != backwardpath[-1] :
                prevp[0] = p[0]
                prevp[1] = p[1]
                dist -= d
            else :
                normx = ((prevp[0] - p[0]) / d)
                normy = ((prevp[1] - p[1]) / d)
                finalx = prevp[0] - normx * dist
                finaly = prevp[1] - normy * dist


    normyaw = np.arctan2(normx, normy)
    newposx = finalx - normy * ret[0]
    newposy = finaly + normy * ret[0]

    cos = np.cos(ret[2] + normyaw)
    sin = np.sin(ret[2] + normyaw)
    return (np.array([[cos, sin, (1 - cos) * 512 - sin * 844 + newposx * 12], [-sin, cos, sin * 512 + (1 - cos) * 844 + newposy * 12]]), 
            [newposx, newposy, ret[2] + normyaw])


def image_diff(M1, M2):
    return np.mean(cv2.divide(cv2.multiply(M1, M2) + 0.1, cv2.multiply(M1, M1) + cv2.multiply(M2, M2) + 0.2)) * 2


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


def network_loc(input_local, input_global):
    with tf.compat.v1.variable_scope('Localize'):

        local_conv1 = tf.compat.v1.layers.conv2d(input_local, 48, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 128 128
        local_conv1 = tf.compat.v1.nn.leaky_relu(local_conv1, alpha=0.1)
        local_conv2 = tf.compat.v1.layers.conv2d(local_conv1, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 64 64
        local_conv2 = tf.compat.v1.nn.leaky_relu(local_conv2, alpha=0.1)
        local_conv3 = tf.compat.v1.layers.conv2d(local_conv2, 96, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 32 32
        local_conv3 = tf.compat.v1.nn.leaky_relu(local_conv3, alpha=0.1)
        local_conv4 = tf.compat.v1.layers.conv2d(local_conv3, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 16 16
        local_conv4 = tf.compat.v1.nn.leaky_relu(local_conv4, alpha=0.1)
        local_conv5 = tf.compat.v1.layers.conv2d(local_conv4, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 8 8 
        local_conv5 = tf.compat.v1.nn.leaky_relu(local_conv5, alpha=0.1)
        local_conv6 = tf.compat.v1.layers.conv2d(local_conv5, 384, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 4 4 
        local_conv6 = tf.compat.v1.nn.leaky_relu(local_conv6, alpha=0.1)
        local_conv7 = tf.compat.v1.layers.conv2d(local_conv6, 512, [4,4], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="valid", use_bias=True) / 25 # 4 4 
        local_conv7 = tf.compat.v1.nn.leaky_relu(local_conv7, alpha=0.1)

        global_conv0 = tf.compat.v1.layers.conv2d(input_global, 32, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 512 512
        global_conv0 = tf.compat.v1.nn.leaky_relu(global_conv0, alpha=0.1)
        global_conv1 = tf.compat.v1.layers.conv2d(global_conv0, 48, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 256 256
        global_conv1 = tf.compat.v1.nn.leaky_relu(global_conv1, alpha=0.1)
        global_conv2 = tf.compat.v1.layers.conv2d(global_conv1, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 9 # 128 128
        global_conv2 = tf.compat.v1.nn.leaky_relu(global_conv2, alpha=0.1)
        global_conv3 = tf.compat.v1.layers.conv2d(global_conv2, 96, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 64 64
        global_conv3 = tf.compat.v1.nn.leaky_relu(global_conv3, alpha=0.1)
        global_conv4 = tf.compat.v1.layers.conv2d(global_conv3, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 16 # 32 32
        global_conv4 = tf.compat.v1.nn.leaky_relu(global_conv4, alpha=0.1)
        global_conv5 = tf.compat.v1.layers.conv2d(global_conv4, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 16 16 
        global_conv5 = tf.compat.v1.nn.leaky_relu(global_conv5, alpha=0.1)
        global_conv6 = tf.compat.v1.layers.conv2d(global_conv5, 384, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 8 8 
        global_conv6 = tf.compat.v1.nn.leaky_relu(global_conv6, alpha=0.1)
        global_conv7 = tf.compat.v1.layers.conv2d(global_conv6, 512, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=True) / 25 # 4 4 
        global_conv7 = tf.compat.v1.nn.leaky_relu(global_conv7, alpha=0.1)
        global_conv8 = tf.compat.v1.layers.conv2d(global_conv7, 512, [4,4], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="valid", use_bias=True) / 25 # 4 4 
        global_conv8 = tf.compat.v1.nn.leaky_relu(global_conv8, alpha=0.1)

        concat = tf.concat([local_conv7, global_conv8], -1)

        flat = tf.compat.v1.layers.Flatten()(concat)

        final_dropout1 = tf.compat.v1.nn.leaky_relu(flat)

        final_dence2 = tf.compat.v1.layers.dense(final_dropout1, 1024, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0), use_bias=True)
        final_dropout2 = tf.compat.v1.nn.leaky_relu(final_dence2)

        final_dence3 = tf.compat.v1.layers.dense(final_dropout1, 256, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0), use_bias=True)
        final_dropout3 = tf.compat.v1.nn.leaky_relu(final_dence3)

        final_dence4 = tf.compat.v1.layers.dense(final_dropout3, 3, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=False)

        return final_dence4 * [0.6, 0.6, 0.1]


tf.compat.v1.disable_eager_execution()
input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])
input_warpedcam = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
input_globalmap = tf.compat.v1.placeholder(tf.float32, [None, 1024, 1024, 3])

output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)

output_loc = network_loc(input_warpedcam, input_globalmap)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.Saver(max_to_keep=0)
saver.restore(sess, "./log_merge_4/log_merge_4_700.ckpt")
#restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
#restorer.restore(sess, "./log_wp_2/model_wp_1700.ckpt")

log_file = open("test_cnn.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

max_time = []
filenames = [x[:-8] for x in os.listdir("/media/user/disk1/20201014_Simulation_Result/") if x[-7:] == "cam.png"]
for i in range(1001):
    s = "%06d" % i
    max_time.append(len([x for x in filenames if x[0:6] == s]))

for index in range(1001):
    xpiece = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(max_time[index]) :
        name = "/media/user/disk1/20201014_Simulation_Result/" + "%06d_%03d" % (index, i)
        cam_image = [cv2.imread(name + "_cam.png").astype(np.float32) / 255.]
        globalmap_image = [cv2.imread(name + "_map.png").astype(np.float32) / 255.]

        forwardpath = []
        backwardpath = []

            
        with open(name + "_arg.txt") as f :
            js = json.load(f)
            forwardpath = js["forward"]
            backwardpath = js["backward"]

            xpiece.append([float(js["x"]), float(js["y"]), float(js["yaw"])])
            xpiece = xpiece[1:]
    

        if i >= 1:
                
            newpiece = [[x[0] - xpiece[2][0], x[1] - xpiece[2][1], x[2] - xpiece[2][2]] for x in xpiece]

            ret_output_unwarp_softmax = sess.run(output_unwarp_softmax, {input_cam:cam_image})
            warpedcam_list = ret_output_unwarp_softmax[:, :, :, 1:]

            ret_output_loc = sess.run(output_loc, {input_warpedcam:warpedcam_list, input_globalmap:globalmap_image})
            print(ret_output_loc)
            (M, ret) = getaffinemaxrix(ret_output_loc[0], forwardpath, backwardpath)


            global_image_patch = globalmap_image[0][492:748, 384:640]
            global_image_norm = cv2.GaussianBlur(global_image_patch, (11, 11), 0)
            global_normed_image = cv2.divide(global_image_patch, global_image_norm + 1.0)
            blur_global_image = cv2.GaussianBlur(global_normed_image, (65, 65), 0)
            _, blur_global_image_th = cv2.threshold(blur_global_image, 0.25, 0.25,cv2.THRESH_TRUNC)
            blur_global_image_th *= 4

            local_image_norm = cv2.GaussianBlur(warpedcam_list[0], (11, 11), 0)
            local_image = cv2.divide(warpedcam_list[0], local_image_norm + 1.0)
            blur_local_image = cv2.GaussianBlur(local_image, (65, 65), 0)
            _, blur_local_image_th = cv2.threshold(blur_local_image,0.25,0.25,cv2.THRESH_TRUNC)
            blur_local_image_th *= 4

            score_o = image_diff(blur_global_image_th, blur_local_image_th)

            global_affined_image = cv2.warpAffine(globalmap_image[0], M, (1024, 1024))
            global_image_patch = global_affined_image[492:748, 384:640]
            global_image_norm = cv2.GaussianBlur(global_image_patch, (11, 11), 0)
            global_normed_image = cv2.divide(global_image_patch, global_image_norm + 1.0)
            blur_global_image = cv2.GaussianBlur(global_normed_image, (65, 65), 0)
            _, blur_global_image_th = cv2.threshold(blur_global_image, 0.25, 0.25,cv2.THRESH_TRUNC)
            blur_global_image_th *= 4


            score_n = image_diff(blur_global_image_th, blur_local_image_th)

            cv2.imshow("cam_image", cam_image[0])
            cv2.imshow("unwarp_image", warpedcam_list[0])
            cv2.imshow("globalmap_image", globalmap_image[0])
            cv2.imshow("map_patch", global_image_patch)

            print(score_o, score_n)

            log_file.write(name[-10:] + "\t" + str(noise_lat) + "\t" + str(noise_lon) + "\t" + str(noise_yaw) + "\t" + \
                str(ret[0]) + "\t" + str(-ret[1]) + "\t" + str(-ret[2]) + "\n" )
                
            cv2.waitKey(1)
