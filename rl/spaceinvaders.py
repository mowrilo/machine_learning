import gym, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def mlp_net(obs):
    hidden1 = tf.tanh(tf.matmul(obs,w['in']) + b['in'])
    hidden2 = tf.tanh(tf.matmul(hidden1,w['h1']) + b['h1'])
    out = tf.nn.softmax(tf.matmul(hidden2,w['h2']) + b['h2'])
    return out


# Returns: rewards, states, actions
def gen_episode(env,render):
    observ = env.reset()
    total_reward = 0
    disc_rate = .9
    pwr = 0
    gts = []
    obss = []
    acts = []
    while True:
        if render:
            env.render()
        obss.append(observ)
        actions = sess.run(net_out, {obs:np.array([observ])})
        cs_actions = np.cumsum(actions)
        roul = np.random.rand()
        chos = 0
        while roul > cs_actions[chos]:
            chos += 1
        acts.append(chos)
        observ,reward,fell,info = env.step(chos)
        gts.append(reward*(disc_rate**pwr))
        if fell:
            print "Died!"
            break
        pwr += 1
    return gts,obss,acts

env = gym.make('SpaceInvaders-ram-v0')
n_outputs = env.action_space.n

w = {'in':tf.Variable(tf.random_uniform([128,80],dtype=tf.float32)),
        'h1':tf.Variable(tf.random_uniform([80,25],dtype=tf.float32)),
        'h2':tf.Variable(tf.random_uniform([25,n_outputs],dtype=tf.float32))}

b = {'in':tf.Variable(tf.random_uniform([1,80],dtype=tf.float32)),
        'h1':tf.Variable(tf.random_uniform([1,25],dtype=tf.float32)),
        'h2':tf.Variable(tf.random_uniform([1,n_outputs],dtype=tf.float32))}
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

obs = tf.placeholder(tf.float32,[None,128])

net_out =  mlp_net(obs)

gt = tf.placeholder(tf.float32,[1,None])
act = tf.placeholder(tf.int32,[1,None])
row = tf.placeholder(tf.int32,[1,None])

learn_rate = .01
print "Gt: ",gt,"\nNetOut: ",tf.gather_nd(net_out,act)
loss = -tf.reduce_sum(gt*tf.log(tf.gather_nd(net_out,tf.stack((row,act),-1))))
opt = tf.train.GradientDescentOptimizer(learn_rate)#GradientDescentOptimizer(learn_rate)
train = opt.minimize(loss)

for count in xrange(1000):
    print "Generating episode number ",count,"..."
    gts,obss,acts = gen_episode(env,False)
    gts = list(reversed(np.cumsum(list(reversed(gts)))))
    rw = np.array([range(0,len(gts))])
    sess.run(train,{gt:np.array([gts]),obs:obss,\
            act:np.array([acts]),row:rw})

_ = raw_input("Move on to test exec?")

test,obss,acts = gen_episode(env,True)
print "Test try got reward = ",sum(test)
