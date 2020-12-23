import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json




batch_size = 16
hidden_size = 128
num_hidden_layers = 3

tf.compat.v1.disable_eager_execution()

input_data = tf.compat.v1.placeholder(tf.float32, [None, 3, 3])
input_time_onehot = tf.compat.v1.placeholder(tf.float32, [None, 3, 3])
input_target = tf.compat.v1.placeholder(tf.float32, [None, 3, 3])


softmax_input_w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[3, hidden_size]), dtype=tf.float32)
softmax_input_b = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[hidden_size]), dtype=tf.float32)

softmax_output_w = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[hidden_size, 3]), dtype=tf.float32)
softmax_output_b = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[3]), dtype=tf.float32)

cells = []
for _ in range(0, num_hidden_layers):
    cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cells.append(cell)

cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

inputs = tf.matmul(input_data, softmax_input_w) + softmax_input_b
outputs_rnn, final_state = tf.compat.v1.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
outputs = tf.matmul(outputs_rnn, softmax_output_w) + softmax_output_b

loss = tf.reduce_sum(tf.math.multiply(tf.square(outputs - input_target), input_time_onehot), axis=1)
optimizer = tf.compat.v1.train.AdamOptimizer(0.001)
operation = optimizer.minimize(loss)

max_time = []
filenames = [x[:-8] for x in os.listdir("/run/user/1000/gvfs/smb-share:server=1.233.226.215,share=data/20201014_Simulation_Result/") if x[-7:] == "cam.png"]
for i in range(1001):
    s = "%06d" % i
    max_time.append(len([x for x in filenames if x[0:6] == s]))
print("max_time", max_time)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
#restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
#restorer.restore(sess, "./log_wp_3/model_wp_2000.ckpt")

saver = tf.compat.v1.train.Saver(max_to_keep=0)
#saver.restore(sess, "./log_dqn_6/log_dqn_6_100.ckpt")
log_file = open("log_rnn_1/log_rnn.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


for epoch in range(1, 100001):
    xbatch = []
    ybatch = []
    onehot_batch = []
    for i in range(16):
        xpiece = []
        onehotpiece = []

        i = 1000
        while max_time[i] < 2:
            i = random.randint(0, 1000)
        start = random.randint(0, max_time[i] - 2)
        for j in range(4):
            if start < max_time[i]:
                name = "%06d_%03d_arg.txt" % (i, start)
                with open("/run/user/1000/gvfs/smb-share:server=1.233.226.215,share=data/20201014_Simulation_Result/" + name) as f :
                    js = json.load(f)
                    xpiece.append([float(js["x"]), float(js["y"]), float(js["yaw"])])
                if start == max_time[i] - 1:
                    onehotpiece.append([0.0, 0.0, 0.0])
                else :
                    onehotpiece.append([0.01, 0.01, 1.0])
            else:
                xpiece.append(xpiece[-1][:])
                onehotpiece.append([0.0, 0.0, 0.0])
            start += 1
        
        r0 =  xpiece[2][0]
        r1 =  xpiece[2][1]
        r2 =  xpiece[2][2]
        for j in range(4):
            xpiece[j][0] -= r0
            xpiece[j][1] -= r1
            xpiece[j][2] -= r2
        xbatch.append(xpiece[0:3])
        ybatch.append(xpiece[1:4])
        onehot_batch.append(onehotpiece[0:3])

    _, _loss = sess.run((operation, loss), {input_data : xbatch, input_target: ybatch, input_time_onehot : onehot_batch})
    print("epoch : " + str(epoch) + " loss : " + str(_loss))
    log_file.write("epoch : " + str(epoch) + " loss : " + str(_loss) + "\n")

    if epoch % 1000 == 0 :
        saver.save(sess, "log_rnn_1/log_rnn_" + str(epoch) + ".ckpt")
