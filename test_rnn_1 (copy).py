import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json





tf.compat.v1.disable_eager_execution()


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

input_rnn_data = tf.compat.v1.placeholder(tf.float32, [None, 1, 3])
input_rnn_target = tf.compat.v1.placeholder(tf.float32, [None, 1, 3])

init_state, output_rnn, cell_rnn, output_rnn_state = network_expect(input_rnn_data)

loss_rnn = tf.reduce_sum(tf.square(output_rnn - input_rnn_target) * [1., 1., 100.])
optimizer_rnn = tf.compat.v1.train.AdamOptimizer(0.001)
operation_rnn = optimizer_rnn.minimize(loss_rnn)

filenames = [x[:-8] for x in os.listdir("/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1_test/") if x[-7:] == "cam.png"]
filetimes = []
for i in range(101):
    s = "%06d" % i
    l = [x for x in filenames if x[0:6] == s]
    l.sort()
    xpiece = []
    if len(l) > 3:
        x = None
        y = None
        yaw = None
        for name in l:
            with open("/media/user/disk1/20201223_Simulation_Result/noise_5.0_0.5+0.5_0.1_test/" + name + "_arg.txt") as f :
                js = json.load(f)
                if x is None:
                    xpiece.append([0., 0., 0.])
                else:
                    xpiece.append([float(js["x"]) - x, float(js["y"]) - y, np.sin(float(js["yaw"]) - yaw)  ])
                x = float(js["x"])
                y = float(js["y"])
                yaw = float(js["yaw"])
        filetimes.append(xpiece)

print(filetimes[1])

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
restorer =  tf.compat.v1.train.Saver()
restorer.restore(sess, "./log_rnn_2/log_rnn_10000.ckpt")

saver = tf.compat.v1.train.Saver(max_to_keep=0)
#saver.restore(sess, "./log_dqn_6/log_dqn_6_100.ckpt")
log_file = open("test_rnn_1.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)

itr = 0

for filelist in filetimes:
    cur_state = sess.run(init_state)
    cur_state = sess.run(output_rnn_state, {input_rnn_data : [[filelist[0]]], init_state : cur_state})
    losssum = 0
    for j in range(2, len(filelist)):
        cur_state, res = sess.run((output_rnn_state, output_rnn), {input_rnn_data : [[filelist[j-1]]], init_state : cur_state})
        print("res : " + str(res))

        log_file.write("test " + str(itr) + "\t" + str(filelist[j][0]) + "\t" + str(filelist[j][1]) + "\t" + str(filelist[j][2]) + "\t" +
            str(res[0][0][0]) + "\t" + str(res[0][0][1]) + "\t" + str(res[0][0][2]) + "\t" + "\n")
