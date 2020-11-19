# coding:utf-8
# 导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

# 基于seed产生随机数
rdm = np.random.RandomState(SEED)
# 随机数返回32行2列的矩阵，表示32组体积和重量作为输入数据集
X = rdm.rand(32,2)
# 从X这个32行2列的矩阵中取出一行，判断如果和小于1给Y赋值1；如果和不小于1给Y赋值0
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n",X)
print("Y_:\n",Y_)

# 定义神经网络的输入、参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))# 用placeholder实现输入定义
y_= tf.placeholder(tf.float32, shape=(None, 1))# 用placeholder实现占位

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))# 正态分布随机数
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))# 正态分布随机数

a = tf.matmul(x, w1)# 点积
y = tf.matmul(a, w2)# 点积

# 定义损失函数及反向传播方法
loss_mse = tf.reduce_mean(tf.square(y-y_)) 
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(loss_mse)
#train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

# 生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()# 初始化
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")
    
    # 训练模型
    STEPS = 3000
    for i in range(STEPS):#3000轮
        start = (i*BATCH_SIZE) % 32 #i*8%32
        end = start + BATCH_SIZE    #i*8%32+8
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))
    
    # 输出训练后的参数取值。
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
# 只搭建承载计算过程
# 计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。 
# 会话（Session）：执行计算图中的节点运算  
    print("w1:\n", w1)
    print("w2:\n", w2)
