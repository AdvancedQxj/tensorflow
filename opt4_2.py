# coding:utf-8
# 酸奶成本1元, 酸奶利润9元
# 预测少了损失大，故不要预测少，故生成的模型会多预测一些
# 导入模块，生成数据集
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455 # 随机种子
COST = 1     # 花费
PROFIT = 9   # 成本

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]

# 定义神经网络的输入、参数和输出，定义前向传播过程
x  = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y  = tf.matmul(x, w1)

# 定义损失函数及反向传播方法
# 定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测
# tf.where:如果condition对应位置值为True那么返回Tensor对应位置为x的值，否则为y的值.
# where(condition, x=None, y=None,name=None)
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end   = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i%500==0:
            print("After %d training steps, w1 is: " % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is: \n", sess.run(w1))
