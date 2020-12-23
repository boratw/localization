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
    return np.array([[cos, sin, cos * ret[0] * 12 + sin * ret[1] * 12 - 512 * (sin + cos - 1)], [-sin, cos, cos * ret[1] * 12 - sin * ret[0] * 12 - 512 * (cos - sin - 1)]])


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
        global_conv7 = tf.compat.v1.nn.leaky_relu(global_conv7, alpha=0.6)


        concat_4 = tf.concat([local_conv4, global_conv5], -1)
        final_conv5 = tf.compat.v1.layers.conv2d(concat_4, 256, [3,3], kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 8 8 
        final_conv5 = tf.compat.v1.nn.leaky_relu(final_conv5, alpha=0.1)
        flat5 = tf.compat.v1.layers.Flatten()(final_conv5)

        concat_5 = tf.concat([local_conv5, global_conv6], -1)
        final_conv6 = tf.compat.v1.layers.conv2d(concat_5, 512, [3,3], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 8 8 
        final_conv6 = tf.compat.v1.nn.leaky_relu(final_conv6, alpha=0.1)
        flat6 = tf.compat.v1.layers.Flatten()(final_conv6)

        concat_6 = tf.concat([local_conv6, global_conv7], -1)
        final_conv7 = tf.compat.v1.layers.conv2d(concat_6, 1024, [3,3], 2, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), padding="same", use_bias=False) / 25 # 8 8 
        final_conv7 = tf.compat.v1.nn.leaky_relu(final_conv7, alpha=0.1)
        flat7 = tf.compat.v1.layers.Flatten()(final_conv7)

        concat_flat = tf.concat([flat5, flat6, flat7], -1)



        final_dropout1 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(concat_flat, rate=0.1))

        final_dence2 = tf.compat.v1.layers.dense(final_dropout1, 1024, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 64
        final_dropout2 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(final_dence2, rate=0.1))

        final_dence3 = tf.compat.v1.layers.dense(final_dropout2, 256, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 16
        final_dropout3 = tf.compat.v1.nn.leaky_relu(tf.compat.v1.layers.dropout(final_dence3, rate=0.1))

        final_output = tf.compat.v1.layers.dense(final_dropout3, 6, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 4

        return final_output


tf.compat.v1.disable_eager_execution()
input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])
input_warpedcam = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
input_globalmap = tf.compat.v1.placeholder(tf.float32, [None, 512, 512, 3])
input_action = tf.compat.v1.placeholder(tf.float32, [None, 6])
input_qvalue = tf.compat.v1.placeholder(tf.float32, [None, 1])
global_step = tf.compat.v1.placeholder(tf.int64)

#input_advantage_resized  = tf.compat.v1.nn.leaky_relu(input_advantage, alpha=5.0)

output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)


output_action = network_loc(input_warpedcam, input_globalmap)

cost_action = tf.square(tf.reduce_sum(tf.math.multiply(input_action, output_action), axis=1, keepdims=True) - input_qvalue)


learning_rate_loc = tf.compat.v1.train.exponential_decay(0.001, global_step, 10, 0.9) 
operation_loc = tf.compat.v1.train.AdamOptimizer(learning_rate_loc).minimize(cost_action)


image_path_list = []
image_path_list.extend(["/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1/" + x[:-8] for x in os.listdir("/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1/") if x[-7:] == "cam.png"])

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
restorer.restore(sess, "./log_wp_3/model_wp_2000.ckpt")

saver = tf.compat.v1.train.Saver(max_to_keep=0)
#saver.restore(sess, "./log_dqn_6/log_dqn_6_100.ckpt")
log_file = open("log_dqn_1/log_dqn_1.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

for epoch in range(1, 3001):
    history_local = []
    history_global = []
    history_action = []
    history_qvalue = []

    for play in range(30) :
        previous_action = -1
        previous_advantage = 0
        realoutput = [0., 0., 0., 0., 0., 0.]
        image_path = random.choice(image_path_list)
        cam_image = [cv2.imread(image_path + "_cam.png").astype(np.float32) / 255.]

        globalmap_image = cv2.imread(image_path + "_map.png").astype(np.float32) / 255.

        if random.random() > 0.5 :
            mirror = True
        else:
            mirror = False
        if mirror :
            cam_image[0] = cv2.flip(cam_image[0], 1)
            globalmap_image = cv2.flip(globalmap_image, 1)

        with open(image_path + "_arg.txt") as f :
            js = json.load(f)

            noise_yaw = -float(js["noise_yaw"])
            rot = float(js["yaw"]) + noise_yaw
            noise_x = float(js["noise_x"])
            noise_y = float(js["noise_y"]) 

            cos = np.cos(-1.570796326794896619 + rot)
            sin = np.sin(-1.570796326794896619 + rot)
            x_noise = -noise_x * cos + noise_y * sin - 9 * np.sin(noise_yaw)
            y_noise = -noise_x * sin - noise_y * cos - 9 * np.cos(noise_yaw)

            if mirror:
                x_noise = -x_noise
                noise_yaw = -noise_yaw


        ret_output_unwarp_softmax = sess.run(output_unwarp_softmax, {input_cam:cam_image})
        warpedcam_list = ret_output_unwarp_softmax[:, :, :, 1:]
        cv2.imshow("unwarp_image", warpedcam_list[0])

        cv2.imshow("original_patch", globalmap_image[384:640, 384:640])

        local_image_norm = cv2.GaussianBlur(warpedcam_list[0], (11, 11), 0)
        local_image = cv2.divide(warpedcam_list[0], local_image_norm + 0.1)
        blur_local_image = cv2.GaussianBlur(local_image, (61, 61), 0)
        #_, blur_local_image_th = cv2.threshold(blur_local_image,0.3333333,0.3333333,cv2.THRESH_TRUNC)
        #blur_local_image_th = blur_local_image_th * 3

        blur_local_image_2 = cv2.multiply(blur_local_image, blur_local_image)
        #blur_local_image_mean = np.mean(blur_local_image_th)

        globalmap_warped_image = globalmap_image
        
        

        localhistory = []
        globalhistory = []
        actionhistory = []
        valuehistory = []
        qvaluehistory = []

        prevmaxscore = 0.
        prevmaxstep = 0

        if random.random() > 0.95:
            while True:
                direction1 = random.randint(0, 5)
                direction2 = random.randint(0, 5)
                if direction1 - direction2 != 3 and direction1 - direction2 != -3:
                    break

        ret2 = [0.0, 0.0, 0.0]
        for step in range(20):
            global_image_norm = cv2.GaussianBlur(globalmap_warped_image[384:640, 384:640], (11, 11), 0)
            global_image_normed = cv2.divide(globalmap_warped_image[384:640, 384:640], global_image_norm + 0.1)
            blur_global_image = cv2.GaussianBlur(global_image_normed, (61, 61), 0)
            #_, blur_global_image_th = cv2.threshold(blur_global_image,0.3333333,0.3333333,cv2.THRESH_TRUNC)
            #blur_global_image_th = blur_global_image_th * 3
                
            score = image_diff(blur_local_image, blur_global_image, blur_local_image_2)

            global_patch = globalmap_warped_image[256:768, 256:768]


            #cv2.imshow("global_map", global_patch)
            #cv2.imshow("global_patch", global_patch[128:384, 128:384])
            #cv2.waitKey(0)


            localhistory.append(warpedcam_list[0])
            globalhistory.append(global_patch)
            valuehistory.append([score])

            ret_output_action = sess.run(output_action, {input_warpedcam:warpedcam_list, input_globalmap:[global_patch]})
            print(ret_output_action)

            maxarg = np.argmax(ret_output_action[0])
            qvaluehistory.append([score + ret_output_action[0][maxarg]])

            if random.random() > 0.7 :
                maxarg = random.randint(0, 5)

            if maxarg == 0:
                ret2[0] += np.cos(ret2[2]) * 0.1
                ret2[1] += np.sin(ret2[2]) * 0.1
            elif maxarg == 1:
                ret2[0] += np.cos(ret2[2]) * -0.1
                ret2[1] += np.sin(ret2[2]) * -0.1
            elif maxarg == 2:
                ret2[0] -= np.sin(ret2[2]) * 0.1
                ret2[1] += np.cos(ret2[2]) * 0.1
            elif maxarg == 3:
                ret2[0] -= np.sin(ret2[2]) * -0.1
                ret2[1] += np.cos(ret2[2]) * -0.1
            elif maxarg == 4:
                ret2[2] += 0.01
            elif maxarg == 5:
                ret2[2] += -0.01

            action = [0., 0., 0., 0., 0., 0.]
            action[maxarg] =  1.0

            actionhistory.append(action)
            

            M = getaffinemaxrix(ret2)
            globalmap_warped_image = cv2.warpAffine(globalmap_image, M, (1024, 1024))


            print("score : ", score)
            print(ret2)
            
            if prevmaxscore * 1.01 < score :
                prevmaxscore = score
                prevmaxstep = step

        if prevmaxstep < 3:
            prevmaxstep = 3
        elif prevmaxstep > 18:
            prevmaxstep = 18
        
        qvaluehistory[prevmaxstep] = valuehistory[prevmaxstep]

        
        history_local.extend(localhistory[:prevmaxstep])
        history_global.extend(globalhistory[:prevmaxstep])
        history_action.extend(actionhistory[:prevmaxstep])
        history_qvalue.extend(qvaluehistory[1:prevmaxstep+1])
        #if original_score < firstscore : 
        #    history_qvalue.extend([[x - falsemove * 0.01 + (firstscore - original_score) / (firstbreak + 1)] for x in maxvaluehistory[1:step]])
        #else:
        #    history_qvalue.extend([[x - falsemove * 0.01 + (firstscore - original_score) * 3 / (firstbreak + 1)] for x in maxvaluehistory[1:step]])




        print("final_reward : ", score)


        log_file.write("epoch " + str(epoch) + " play " + str(play) + " final score : " + str(score) + "\n")


        cv2.imshow("global_map", globalhistory[prevmaxstep])
        cv2.imshow("global_patch", globalhistory[prevmaxstep][128:384, 128:384])

        cv2.waitKey(10)


    for play in range(100) :
        dic = random.sample(range(len(history_local)), k=8)

        batch_local = [history_local[r] for r in dic]
        batch_global = [history_global[r] for r in dic]
        batch_action = [history_action[r] for r in dic]
        batch_qvalue = [history_qvalue[r] for r in dic]


        _, ret2 =  sess.run((operation_loc, cost_action), {input_warpedcam:batch_local, input_globalmap:batch_global,
            input_action:batch_action, input_qvalue:batch_qvalue, global_step:epoch})
        
        print("cost : ", ret2)
        log_file.write("epoch " + str(epoch) + " play " + str(play) + " cost : " + str(ret2) + "\n")

    if epoch % 100 == 0 :
        saver.save(sess, "log_dqn_1/log_dqn_1_" + str(epoch) + ".ckpt")
