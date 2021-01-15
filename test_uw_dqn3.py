import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
import numpy as np
import cv2
import random
import json

def getaffinemaxrix(ret):
    cos = np.cos(ret[2])
    sin = np.sin(ret[2])
    return np.array([[cos, sin, ret[0] * 24 - 960 * sin - 512 * cos + 512],
         [-sin, cos, ret[1] * 24 - 960 * cos + 512 * sin + 960]])

def image_diff(M1, M2, M1_2):
    #return (np.mean(cv2.multiply(M1, M2)) / np.mean(M1)) ** 2 * 100.
    #score = np.mean(cv2.divide(cv2.multiply(M1, M2) + 0.1, M1_2 + cv2.multiply(M2, M2) + 0.2)) * 2
    score = np.mean(cv2.divide(cv2.multiply(M1, M2) + 0.1, M1_2 + cv2.multiply(M2, M2) + 0.2)) * 2
    return (score ** 8) * 10.


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

        local_conv1 = tf.compat.v1.layers.conv2d(input_local, 32, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 9 # 128 128
        local_conv1 = tf.compat.v1.nn.leaky_relu(local_conv1, alpha=0.1)
        local_conv2 = tf.compat.v1.layers.conv2d(local_conv1, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 9 # 64 64
        local_conv2 = tf.compat.v1.nn.leaky_relu(local_conv2, alpha=0.1)
        local_conv3 = tf.compat.v1.layers.conv2d(local_conv2, 96, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 16 # 32 32
        local_conv3 = tf.compat.v1.nn.leaky_relu(local_conv3, alpha=0.1)
        local_conv4 = tf.compat.v1.layers.conv2d(local_conv3, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 16 # 16 16
        local_conv4 = tf.compat.v1.nn.leaky_relu(local_conv4, alpha=0.1)
        local_conv5 = tf.compat.v1.layers.conv2d(local_conv4, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 8 8 
        local_conv5 = tf.compat.v1.nn.leaky_relu(local_conv5, alpha=0.1)
        local_conv6 = tf.compat.v1.layers.conv2d(local_conv5, 512, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 4 4 
        local_conv6 = tf.compat.v1.nn.leaky_relu(local_conv6, alpha=0.1)

        global_conv1 = tf.compat.v1.layers.conv2d(input_global, 32, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 9 # 256 256
        global_conv1 = tf.compat.v1.nn.leaky_relu(global_conv1, alpha=0.1)
        global_conv2 = tf.compat.v1.layers.conv2d(global_conv1, 64, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 9 # 128 128
        global_conv2 = tf.compat.v1.nn.leaky_relu(global_conv2, alpha=0.1)
        global_conv3 = tf.compat.v1.layers.conv2d(global_conv2, 96, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 16 # 64 64
        global_conv3 = tf.compat.v1.nn.leaky_relu(global_conv3, alpha=0.1)
        global_conv4 = tf.compat.v1.layers.conv2d(global_conv3, 128, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 16 # 32 32
        global_conv4 = tf.compat.v1.nn.leaky_relu(global_conv4, alpha=0.1)
        global_conv5 = tf.compat.v1.layers.conv2d(global_conv4, 256, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 16 16 
        global_conv5 = tf.compat.v1.nn.leaky_relu(global_conv5, alpha=0.1)
        global_conv6 = tf.compat.v1.layers.conv2d(global_conv5, 512, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 8 8 
        global_conv6 = tf.compat.v1.nn.leaky_relu(global_conv6, alpha=0.1)
        global_conv7 = tf.compat.v1.layers.conv2d(global_conv6, 1024, [5,5], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 4 4 
        global_conv7 = tf.compat.v1.nn.leaky_relu(global_conv7, alpha=0.1)

        flat_local = tf.compat.v1.layers.Flatten()(local_conv6)
        flat_global = tf.compat.v1.layers.Flatten()(global_conv7)

        concat_flat = tf.concat([flat_local, flat_global], -1)



        final_dropout1 = tf.compat.v1.nn.leaky_relu(concat_flat)

        final_dence2 = tf.compat.v1.layers.dense(final_dropout1, 1024, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 64
        final_dropout2 = tf.compat.v1.nn.leaky_relu(final_dence2)

        final_dence3 = tf.compat.v1.layers.dense(final_dropout2, 256, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 16
        final_dropout3 = tf.compat.v1.nn.leaky_relu(final_dence3)

        final_output = tf.compat.v1.layers.dense(final_dropout3, 4, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 4

        return final_output


tf.compat.v1.disable_eager_execution()
input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])
input_warpedcam = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
input_globalmap = tf.compat.v1.placeholder(tf.float32, [None, 512, 512, 3])

output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)

output_action = network_loc(input_warpedcam, input_globalmap)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
saver = tf.compat.v1.train.Saver(max_to_keep=0)
saver.restore(sess, "./log_dqn_3/log_dqn_3_2000.ckpt")
#restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
#restorer.restore(sess, "./log_wp_2/model_wp_1700.ckpt")

log_file = open("test_dqn3_1.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


image_path_list = []
image_path_list.extend(["/media/user/disk1/20201223_Simulation_Result/noise_no_y_0.3_0.06_test/" + x[:-8] for x in os.listdir("/media/user/disk1/20201223_Simulation_Result/noise_no_y_0.3_0.06_test/") if x[-8:] == "_cam.png"])


for image_path in image_path_list:
    cam_image = [cv2.imread(image_path + "_cam.png").astype(np.float32) / 255.]
    globalmap_image = cv2.imread(image_path + "_map.png").astype(np.float32) / 255.

    with open(image_path + "_arg.txt") as f :
        js = json.load(f)

        noise_lat = float(js["noise_lat"])
        noise_yaw = float(js["noise_yaw"])


   
    ret_output_unwarp_softmax = sess.run(output_unwarp_softmax, {input_cam:cam_image})
    warpedcam_list = ret_output_unwarp_softmax[:, :, :, 1:]

    cv2.imshow("cam_image", cam_image[0])
    cv2.imshow("unwarp_image", warpedcam_list[0])

    local_image_orig = cv2.resize(warpedcam_list[0], (512, 512))[128:384, 128:384]
    local_image_norm = cv2.GaussianBlur(local_image_orig, (11, 11), 0)
    local_image = cv2.divide(local_image_orig, local_image_norm + 0.1)
    blur_local_image = cv2.GaussianBlur(local_image, (61, 61), 0)

    blur_local_image_2 = cv2.multiply(blur_local_image, blur_local_image)


    globalmap_warped_image = globalmap_image

    blur_global_image = cv2.GaussianBlur(globalmap_warped_image[492:748, 384:640], (61, 61), 0)

    score_o = image_diff(blur_local_image, blur_global_image, blur_local_image_2)
    
    maxscore = score_o * 0.95
    maxret = [0., 0., 0.]

    ret2 = [0.0, 0.0, 0.0]
    for step in range(20):
        global_patch = globalmap_warped_image[256:768, 256:768]
        cv2.imshow("global_patch", global_patch)
        ret_output_action = sess.run( output_action, {input_warpedcam:warpedcam_list, input_globalmap:[global_patch]})
        print(ret_output_action)


        maxarg = np.argmax(ret_output_action[0])

        if maxarg == 0:
            ret2[0] += 0.1
        elif maxarg == 1:
            ret2[0] -= 0.1
        elif maxarg == 2:
            ret2[2] += 0.01
        elif maxarg == 3:
            ret2[2] -= 0.01

        M = getaffinemaxrix(ret2)
        globalmap_warped_image = cv2.warpAffine(globalmap_image, M, (1024, 1024))


        blur_global_image = cv2.GaussianBlur(globalmap_warped_image[384:640, 384:640], (61, 61), 0)
        score = image_diff(blur_local_image_2, blur_global_image, blur_local_image_2)
        if maxscore < score:
            maxscore = score
            maxret[0] = ret2[0]
            maxret[1] = ret2[1]
            maxret[2] = ret2[2]

        

        blur_global_image = cv2.GaussianBlur(globalmap_warped_image[384:640, 384:640], (61, 61), 0)
                
        score = image_diff(blur_local_image_2, blur_global_image, blur_local_image_2)
        print("score : ", score)

        cv2.imshow("blur_local_image_th", blur_local_image_2)
        cv2.imshow("blur_global_image_th", blur_global_image)


        cv2.imshow("map_patch", globalmap_warped_image[384:640, 384:640])

        globalmap_copied_image = globalmap_image.copy()

        cpx1 = 384 - ret2[0] * 12 - 512
        cpx2 = 640 - ret2[0] * 12  - 512
        cpx3 = -ret2[0] * 12 
        cpy1 = 384 - ret2[1] * 12  - 512
        cpy2 = 640 - ret2[1] * 12  - 512
        cpy3 = -ret2[1] * 12 

        cos = np.cos(-ret2[2])
        sin = np.sin(-ret2[2])

        tpx1 = cpx1 * cos + cpy1 * sin + 512
        tpy1 = -cpx1 * sin + cpy1 * cos + 512
        tpx2 = cpx1 * cos + cpy2 * sin + 512
        tpy2 = -cpx1 * sin + cpy2 * cos + 512
        tpx3 = cpx2 * cos + cpy2 * sin + 512
        tpy3 = -cpx2 * sin + cpy2 * cos + 512
        tpx4 = cpx2 * cos + cpy1 * sin + 512
        tpy4 = -cpx2 * sin + cpy1 * cos + 512
        outx = cpx3 * cos + cpy3 * sin + 512
        outy = -cpx3 * sin + cpy3 * cos + 512

        pts = np.array([[[tpx1, tpy1], [tpx2, tpy2], [tpx3, tpy3], [tpx4, tpy4]]], np.int32)
        cv2.polylines(globalmap_copied_image, pts, True, (1., 1., 1), 1)
        cv2.line(globalmap_copied_image, (int((tpx1 + tpx2 + tpx3 + tpx4) / 4), int((tpy1 + tpy2 + tpy3 + tpy4) / 4)), ( int((tpx2 + tpx3) / 2), int((tpy2 + tpy3) / 2)),  (1., 1., 1.), 1)

        cv2.imshow("globalmap_copied_image", globalmap_copied_image)

        cv2.waitKey(1)
        if ret_output_action[0][maxarg] < 0.:
            break
        
    
    #ret_o = gettrajaffinematrix((noise_lat, noise_lon, noise_yaw), forwardpath, backwardpath)
    #ret_m = gettrajaffinematrix(maxret, forwardpath, backwardpath)
    #ret_r = gettrajaffinematrix(ret2, forwardpath, backwardpath)

    #print("gt : ", noise_lat, noise_lon, -noise_yaw)
    print("ret2 : ", ret2)


    log_file.write("result : %f\t%f\t%f\t%f\t%f\t%f\n" % 
        (noise_lat, -noise_yaw, score_o, maxret[0] * np.cos(maxret[2]), maxret[2], maxscore))