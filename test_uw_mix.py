import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json





tf.compat.v1.disable_eager_execution()


def drawmap(img, x, y, rot) :
    color = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    cos = np.cos(-1.570796327 - rot)
    sin = np.sin(-1.570796327 - rot)
    diff = [0., 0., 0.]
    for lane in lines:
        if not (lane[2] > x + 128 or lane[3] < x - 128 or lane[4] > y + 128 or lane[5] < y - 128) :
            line = np.array([ [((pt[0] - x) * cos - (pt[1] - y) * sin) * -12 + 512, ((pt[0] - x) * sin + (pt[1] - y) * cos) * 12 + 736] for pt in lane[1] if -128 < (pt[0] - x) < 128 and -128 < (pt[1] - y) < 128 ], np.int32)
            if len(line) >= 2:
                cv2.polylines(img, [line], False, color[lane[0]], 2)

    
def getaffinemaxrix(ret):
    cos = np.cos(ret[2])
    sin = np.sin(ret[2])
    return np.array([[cos, sin, ret[0] * 12 - 736 * sin - 512 * cos + 512],
         [-sin, cos, ret[1] * 12 - 736 * cos + 512 * sin + 736]])

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

        final_output = tf.compat.v1.layers.dense(final_dropout3, 6, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.5), use_bias=True) / 4

        return final_output


def network_expect(input_data):
    with tf.compat.v1.variable_scope('Expect'):

        batch_size = 1
        hidden_size = 128
        num_hidden_layers = 2
        
        cells = []
        for _ in range(0, num_hidden_layers):
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size), output_keep_prob=0.9)
            cells.append(cell)

        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)

        w_in = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([3, 128], stddev=0.1), trainable=True, name="w_in")
        w_out = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([128, 3], stddev=0.1), trainable=True, name="w_out")

        inputs = tf.matmul(input_data, w_in)
        outputs_rnn, final_state = tf.compat.v1.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
        outputs = tf.matmul(outputs_rnn, w_out)

        return initial_state, outputs, cell, final_state

input_cam = tf.compat.v1.placeholder(tf.float32, [None, 256, 512, 3])
input_warpedcam = tf.compat.v1.placeholder(tf.float32, [None, 256, 256, 3])
input_globalmap = tf.compat.v1.placeholder(tf.float32, [None, 512, 512, 3])

output_unwarp = network_uw(input_cam)
output_unwarp_softmax = tf.compat.v1.nn.softmax(output_unwarp)
output_action = network_loc(input_warpedcam, input_globalmap)

input_rnn_data = tf.compat.v1.placeholder(tf.float32, [None, 1, 3])
input_rnn_target = tf.compat.v1.placeholder(tf.float32, [None, 1, 3])

init_state, output_rnn, cell_rnn, output_rnn_state = network_expect(input_rnn_data)

image_path = "/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1_test/"

filenames = [x[:-8] for x in os.listdir(image_path) if x[-7:] == "cam.png"]
filetimes = []
for i in range(101):
    s = "%06d" % i
    l = [x for x in filenames if x[0:6] == s]
    l.sort()
    xpiece = []
    if len(l) > 3:
        filetimes.append(l)

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
        

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

saver = tf.compat.v1.train.Saver(max_to_keep=0)

restorer1 =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name or 'Localize' in v.name])
restorer1.restore(sess, "./log_dqn_3/log_dqn_3_2000.ckpt")

restorer2 =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Expect' in v.name])
restorer2.restore(sess, "./log_rnn_2/log_rnn_10000.ckpt")

log_file = open("test_mix_1.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

itr = 0

for filelist in filetimes:
    cur_state = sess.run(init_state)
    cur_state = sess.run(output_rnn_state, {input_rnn_data : [[filelist[0]]], init_state : cur_state})
    
    with open(image_path + filelist[0] + "_arg.txt") as f :
        js = json.load(f)

        noise_x = float(js["noise_x"])
        noise_y = float(js["noise_y"])
        yaw = float(js["yaw"])
        noise_yaw = float(js["noise_yaw"])
        state = [float(js["x"]), float(js["y"]), yaw]

        cos = np.cos(1.570796327 + yaw - noise_yaw)
        sin = np.sin(1.570796327 + yaw - noise_yaw)
        noise_lat = noise_x * cos + noise_y * sin
        noise_lon = - noise_x * sin + noise_y * cos




    losssum = 0
    for j in range(len(1, filelist)):
        cur_state, res = sess.run((output_rnn_state, output_rnn), {input_rnn_data : [[filelist[j]]], init_state : cur_state})
        print("res : " + str(res))


        log_file.write("test " + str(itr) + "\t" + str(filelist[j][0]) + "\t" + str(filelist[j][1]) + "\t" + str(filelist[j][2]) + "\t" +
            str(res[0][0][0]) + "\t" + str(res[0][0][1]) + "\t" + str(res[0][0][2]) + "\t" + "\n")
