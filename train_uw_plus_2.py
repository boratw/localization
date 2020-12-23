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
    return np.array([[cos, sin, (1 - cos) * 512 - sin * 844 + newposx * 12], [-sin, cos, sin * 512 + (1 - cos) * 844 + newposy * 12]])


def image_diff(M1, M2, M1_2):
    return np.mean(cv2.divide(cv2.multiply(M1, M2) + 0.1, M1_2 + cv2.multiply(M2, M2) + 0.5)) * 2


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

        final_dropout1 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(flat, rate=0.1))

        final_dence2 = tf.compat.v1.layers.dense(final_dropout1, 1024, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0), use_bias=True)
        final_dropout2 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(final_dence2, rate=0.1))

        final_dence3 = tf.compat.v1.layers.dense(final_dropout1, 256, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=1.0), use_bias=True)
        final_dropout3 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(final_dence3, rate=0.1))

        final_dence4 = tf.compat.v1.layers.dense(final_dropout3, 3, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=False)

        return final_dence4 * [0.6, 0.6, 0.1]


tf.compat.v1.disable_eager_execution()
input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])
input_warpedcam = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
input_globalmap = tf.compat.v1.placeholder(tf.float32, [None, 1024, 1024, 3])
input_gt_loc = tf.compat.v1.placeholder(tf.float32, [None, 3])
global_step = tf.compat.v1.placeholder(tf.int64)


output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)


output_loc = network_loc(input_warpedcam, input_globalmap)

cost_loc = tf.reduce_sum(tf.square(input_gt_loc - output_loc))
learning_rate_loc = tf.compat.v1.train.exponential_decay(0.001, global_step, 10, 0.9) 
operation_loc = tf.compat.v1.train.AdamOptimizer(learning_rate_loc).minimize(cost_loc)

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

directory = "/mnt/Localization/20200702_Simulation_Result/3.0/"

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
restorer.restore(sess, "./log_wp_2/model_wp_360.ckpt")

saver = tf.compat.v1.train.Saver(max_to_keep=0)
#saver.restore(sess, "./log_wp_1/model_wp_520.ckpt")
log_file = open("log_merge_4/log_merge_4.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

diff = [0.5, 0.5, 0.05]
moving_average = [0., 0., 0.]

for epoch in range(1, 1001):
    for step in range(128) :
        log_file.write("epoch " + str(epoch) + " step " + str(step) + " : ")
        dic = random.sample(range(30000), k=8)
        seed = random.randint(0, 65535)
        tf.compat.v1.random.set_random_seed(seed)
        cam_image = [cv2.imread(directory + "%06d_cam.png" % r).astype(np.float32) / 255. for r in dic]

        globalmap_image = [cv2.imread(directory + "%06d_map.png" % r).astype(np.float32) / 255. for r in dic]
        forwardpath = []
        backwardpath = []

        for r in range(8):
            if random.random() > 0.5 :
                mirror = True
            else:
                mirror = False
            if mirror :
                cam_image[r] = cv2.flip(cam_image[r], 1)
                globalmap_image[r] = cv2.flip(globalmap_image[r], 1)
            with open(directory + "%06d_arg.txt" % dic[r]) as f :
                js = json.load(f)
                if mirror :
                    forwardpath.append([[-x[0], x[1]] for x in js["forward"]])
                    backwardpath.append([[-x[0], x[1]] for x in js["backward"]])
                else :
                    forwardpath.append(js["forward"])
                    backwardpath.append(js["backward"])

        ret_output_unwarp_softmax = sess.run(output_unwarp_softmax, {input_cam:cam_image})
        warpedcam_list = ret_output_unwarp_softmax[:, :, :, 1:]
        cv2.imshow("cam_image", cam_image[0])
        #cv2.imshow("map_image", globalmap_image[0])
        cv2.imshow("unwarp_image", warpedcam_list[0])

        tf.compat.v1.random.set_random_seed(seed)
        ret_output_loc = sess.run(output_loc, {input_warpedcam:warpedcam_list, input_globalmap:globalmap_image})
        print("loc_output : ", ret_output_loc[0])
        differ = ret_output_loc.copy()
        for r in range(8):
            blur_global_image = cv2.GaussianBlur(globalmap_image[r], (65, 65), 0)
            _, blur_global_image_th = cv2.threshold(blur_global_image,0.25,0.25,cv2.THRESH_TRUNC)
            blur_global_image_th *= 4

            local_image_norm = cv2.GaussianBlur(warpedcam_list[r], (11, 11), 0)
            local_image = cv2.divide(warpedcam_list[r], local_image_norm + 1.0)
            blur_local_image = cv2.GaussianBlur(local_image, (65, 65), 0)
            _, blur_local_image_th = cv2.threshold(blur_local_image,0.2,0.2,cv2.THRESH_TRUNC)
            blur_local_image_th *= 5
            blur_local_image_2 = cv2.multiply(blur_local_image_th, blur_local_image_th)

            
            if r == 0:
                cv2.imshow("blur_local_image", blur_local_image_th)
                cv2.imshow("blur_global_image", blur_global_image_th)

            score_o = image_diff(blur_local_image_th, blur_global_image_th[492:748, 384:640], blur_local_image_2)
            if r == 0:
                M = getaffinemaxrix(ret_output_loc[r], forwardpath[r], backwardpath[r])
                blur_global_image2 = cv2.warpAffine(blur_global_image_th, M, (1024, 1024))
                cv2.imshow("blur_ans_image", blur_global_image2)
            max_score = 0.

            for i in [0, 1, 2] :
                ret2 = ret_output_loc[r].copy()
                ret2[i] += diff[i]

                M = getaffinemaxrix(ret2, forwardpath[r], backwardpath[r])
                blur_global_image2 = cv2.warpAffine(blur_global_image_th, M, (1024, 1024))

                score_p = image_diff(blur_local_image_th, blur_global_image2[492:748, 384:640], blur_local_image_2)

                ret2[i] -= diff[i] * 2

                M = getaffinemaxrix(ret2, forwardpath[r], backwardpath[r])
                blur_global_image2 = cv2.warpAffine(blur_global_image_th, M, (1024, 1024))

                score_n = image_diff(blur_local_image_th, blur_global_image2[492:748, 384:640], blur_local_image_2)


                score = (score_p - score_n) * 100
                if max_score < score_p :
                    max_score = score_p
                if max_score < score_n :
                    max_score = score_n

                if moving_average[i] * score < 0 :
                    if np.abs(score) > diff[i] * 4:
                        if score > 0 :
                            differ[r][i] += diff[i] * 6
                        else :
                            differ[r][i] -= diff[i] * 6
                    else :
                        differ[r][i] += score * 1.5
                else:
                    if np.abs(score) > diff[i] * 4:
                        if score > 0 :
                            differ[r][i] += diff[i] * 4
                        else :
                            differ[r][i] -= diff[i] * 4
                    else :
                        differ[r][i] += score
            
        
            if max_score < score_o * 0.98 :
                print("max_score", max_score)
                differ[r][0] = 0.0
                differ[r][1] = 0.0
                differ[r][2] = 0.0

        meanret = np.mean(ret_output_loc, axis=0)
        for i in range(3):
            moving_average[i] = moving_average[i] * 0.5 + meanret[i] * 0.5

        tf.compat.v1.random.set_random_seed(seed)
        _, ret  = sess.run((operation_loc, cost_loc), {input_warpedcam:warpedcam_list, input_globalmap:globalmap_image, input_gt_loc:differ, global_step:epoch})
        print("loc_gt : ", differ[0])
        print("cost of localize : ", ret)
        print("moving_average : ", moving_average)
        log_file.write(str(ret) + "\n")
        cv2.waitKey(10)


    if epoch % 20 == 0:
        saver.save(sess, "log_merge_4/log_merge_4_" + str(epoch) + ".ckpt")
