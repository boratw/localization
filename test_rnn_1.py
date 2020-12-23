import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
import cv2
import random
import json




batch_size = 1
hidden_size = 128
num_hidden_layers = 3

tf.compat.v1.disable_eager_execution()

input_data = tf.compat.v1.placeholder(tf.float32, [None, 3, 3])


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


max_time = []
filenames = [x[:-8] for x in os.listdir("/run/user/1000/gvfs/smb-share:server=1.233.226.215,share=data/20201014_Simulation_Result/") if x[-7:] == "cam.png"]
for i in range(1001):
    s = "%06d" % i
    max_time.append(len([x for x in filenames if x[0:6] == s]))

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
#restorer =  tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.trainable_variables() if 'Unwarp' in v.name])
#restorer.restore(sess, "./log_wp_3/model_wp_2000.ckpt")

saver = tf.compat.v1.train.Saver(max_to_keep=0)
saver.restore(sess, "./log_rnn_1/log_rnn_100000.ckpt")
log_file = open("test_rnn.txt", "wt")

np.set_printoptions(suppress=True)
np.set_printoptions(precision=8)


for index in range(1001):
    xpiece = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(max_time[index]) :
        name = "/run/user/1000/gvfs/smb-share:server=1.233.226.215,share=data/20201014_Simulation_Result/" + "%06d_%03d" % (index, i)
        image = cv2.imread(name + "_map.png").astype(np.float32) / 255.
        image = image[512:, 256:768]
        with open(name + "_arg.txt") as f :
            js = json.load(f)
            xpiece.append([float(js["x"]), float(js["y"]), float(js["yaw"])])
            xpiece = xpiece[1:]
        
        cos = np.cos(xpiece[2][2] + 1.57079632679)
        sin = np.sin(xpiece[2][2] + 1.57079632679)

        newpiece = [[x[0] - xpiece[2][0], x[1] - xpiece[2][1], x[2] - xpiece[2][2]] for x in xpiece]

        if i >= 1:
            cv2.circle(image, (256, 256), 12, (1., 1., 0.), 2)
            cv2.circle(image, (256, 256), 6, (1., 1., 0.), -1)

            x = int(((newpiece[3][0]) * cos + (newpiece[3][1]) * sin) * -12) + 256
            y = int(((newpiece[3][0]) * -sin + (newpiece[3][1]) * cos) * 12) + 256
            cv2.circle(image, (x, y), 12, (1., 1., 0.), 2)
            cv2.circle(image, (x, y), 6, (1., 1., 0.), -1)

        if i >= 2:
            x = int(((newpiece[1][0]) * cos + (newpiece[1][1]) * sin) * -12) + 256
            y = int(((newpiece[1][0]) * -sin + (newpiece[1][1]) * cos) * 12) + 256
            cv2.circle(image, (x, y), 12, (1., 1., 0.), 2)
            cv2.circle(image, (x, y), 6, (1., 1., 0.), -1)
            cv2.line(image, (x, y), (256, 256), (1., 1., 0.), 2)

        if i >= 3:
            x2 = int(((newpiece[0][0]) * cos + (newpiece[0][1]) * sin) * -12) + 256
            y2 = int(((newpiece[0][0]) * -sin + (newpiece[0][1]) * cos) * 12) + 256
            cv2.circle(image, (x2, y2), 12, (1., 1., 0.), 2)
            cv2.circle(image, (x2, y2), 6, (1., 1., 0.), -1)
            cv2.line(image, (x, y), (x2, y2), (1., 1., 0.), 2)

            _outputs = sess.run(outputs, {input_data : [newpiece[:3]]})

            x = int(((_outputs[0][2][0]) * cos + (_outputs[0][2][1]) * sin) * -12) + 256
            y = int(((_outputs[0][2][0]) * -sin + (_outputs[0][2][1]) * cos) * 12) + 256
            cv2.circle(image, (x, y), 12, (0., 1., 1.), 2)
            cv2.circle(image, (x, y), 6, (0., 1., 1.), -1)

            print(name)
            print(_outputs)
            log_file.write(name[-10:] + "\t" + str(newpiece[3][0]) + "\t" + str(newpiece[3][1]) + "\t" + str(newpiece[3][2]) + "\t" + 
                 str(_outputs[0][2][0]) + "\t" + str(_outputs[0][2][1]) + "\t" + str(_outputs[0][2][2]) + "\n")

            cv2.imshow("image", image)
            cv2.waitKey(500)