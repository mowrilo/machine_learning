import gym, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def mlp_net(obs):
    hidden1 = tf.nn.relu(tf.matmul(obs,w[0]) + b[0])
    hidden2 = tf.nn.relu(tf.matmul(hidden1,w[1]) + b[1])
    logits = tf.sigmoid(tf.matmul(hidden2,w[2]) + b[2])
    out = tf.nn.softmax(logits)
    return out[0]


# Returns: rewards, states, actions
def gen_episode(env,render):
    observ = env.reset()
    total_reward = 0
    gts = []
    obss = []
    acts = []
    while True:
        if render:
            env.render()
        obss.append(obss)
        actions = sess.run(net_out, {obs:np.array([observ])})
        cs_actions = np.cumsum(actions)
        roul = np.random.rand()
        chos = 0
        while roul > cs_actions[chos]:
            chos += 1
        acts.append(chos)
        observ,reward,fell,info = env.step(chos)
        gts.append(reward)
        if fell:
            print "Died!"
            break
    return gts,obss,acts

env = gym.make('SpaceInvaders-ram-v0')
n_outputs = env.action_space.n

w = [tf.Variable(tf.random_uniform([128,80],dtype=tf.float32)),
        tf.Variable(tf.random_uniform([80,20],dtype=tf.float32)),
        tf.Variable(tf.random_uniform([20,n_outputs],dtype=tf.float32))]

b = [tf.Variable(tf.random_uniform([1,80],dtype=tf.float32)),
        tf.Variable(tf.random_uniform([1,20],dtype=tf.float32)),
        tf.Variable(tf.random_uniform([1,n_outputs],dtype=tf.float32))]
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

obs = tf.placeholder(tf.float32,[None,128])
y = tf.placeholder(tf.float32,[None,1])

net_out =  mlp_net(obs)

alpha = tf.placeholder(tf.float32,shape=())
gt = tf.placeholder(tf.float32,shape=())
act = tf.placeholder(tf.int32,shape=())

grads_w = [tf.gradients(net_out[act],w[0])[0],
        tf.gradients(net_out[act],w[1])[0],
        tf.gradients(net_out[act],w[2])[0]]
grads_b = [tf.gradients(net_out[act],b[0])[0],
        tf.gradients(net_out[act],b[1])[0],
        tf.gradients(net_out[act],b[2])[0]]

w_updates = [w[0].assign(w[0] + (alpha*gt*grads_w[0]/net_out[act])),
        w[1].assign(w[1] + (alpha*gt*grads_w[1]/net_out[act])),
        w[2].assign(w[2] + (alpha*gt*grads_w[2]/net_out[act]))]
b_updates = [b[0].assign(b[0] + (alpha*gt*grads_b[0]/net_out[act])),
        b[1].assign(b[1] + (alpha*gt*grads_b[1]/net_out[act])),
        b[2].assign(b[2] + (alpha*gt*grads_b[2]/net_out[act]))]

learn_rate = .1

for count in xrange(200):
    print "Generating episode number ",count,"..."
    gts,obss,acts = gen_episode(env,False)
    gts = list(reversed(np.cumsum(list(reversed(gts)))))
    for i in xrange(len(gts)):
        this_gt = gts[i]
        this_obs = obss[i]
        this_act = acts[i]
        for k in xrange(2,-1,-1):
            print "Gt: ",this_gt
            sess.run(w_updates[k],{alpha:learn_rate,gt:this_gt,obs:this_obs,act:this_act})
            sess.run(b_updates[k],{alpha:learn_rate,gt:this_gt,obs:this_obs,act:this_act})

test,obss,acts = gen_episode(env,True)
print "Test try got reward = ",sum(test)
