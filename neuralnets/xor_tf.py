import numpy as np
import tensorflow as tf

def mlp_eval(x):
    hidden = (tf.matmul(x,w['l1']) + b['l1'])#tf.nn.relu
    out = tf.sigmoid(tf.matmul(hidden,w['l2']) + b['l2'])
    return out

sess = tf.Session()

dataX = np.array([[0,0],[0,1],[1,0],[1,1]])
dataY = np.array([[0,1,1,0]]).transpose()
learning_rate = .1

x = tf.placeholder(tf.float32,[None,2])
y = tf.placeholder(tf.float32,[None,1])

w = {'l1':tf.Variable(tf.random_uniform([2,2]),dtype=tf.float32),
        'l2':tf.Variable(tf.random_uniform([2,1]),dtype=tf.float32)}
b = {'l1':tf.Variable(tf.random_uniform([2]),dtype=tf.float32),
        'l2':tf.Variable(tf.random_uniform([1]),dtype=tf.float32)}

init = tf.global_variables_initializer()
sess.run(init)

net_out = mlp_eval(x)
pred = tf.cast(net_out > .5, tf.int32)
cost = tf.reduce_sum(tf.square(y - net_out))
opt = tf.train.GradientDescentOptimizer(learning_rate)
train = opt.minimize(cost)

print "Training..."
for i in xrange(10000):
    sess.run(train,{x:dataX,y:dataY})
print "Done!"

predictions = sess.run(pred,{x:dataX})
print "True values:\n",dataY
print "Predicted values:\n",predictions
