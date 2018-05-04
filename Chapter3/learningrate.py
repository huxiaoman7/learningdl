import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

x_data = np.random.rand(50).astype(np.float32)
y_data = x_data * 0.1 + 0.3;

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss=tf.reduce_mean(tf.square(y-y_data))

global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

init = tf.global_variables_initializer()
###

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

plt.ion()
plt.show()


for step in range(300):
    sess.run(learning_step)
    if step % 20 == 0:
        y_value=sess.run(y)
        ax.scatter(x_data,y_data)
        ax.scatter(x_data,y_value)
        plt.pause(1)
