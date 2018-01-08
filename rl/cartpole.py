# simple hill climbing reinforcement learning algorithm
# to learn to balance a pole on a cart (cartpole problem)
# author: Murilo V. F. Menezes

import gym
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def eval(params,env,render):
    obs = env.reset()
    total_reward = 0
    for ts in xrange(500):
        if render:
            env.render()
        if np.dot(obs,params) < 0:
            action=0
        else:
            action=1
        obs,reward,fell,info = env.step(action)
        total_reward += reward
        if fell:
            #print "Fell!"
            break
    #print ts
    return total_reward

env = gym.make('CartPole-v0')
params = np.random.rand(4) - .5

first_try = eval(params,env,True)
print "First try got to t = " + str(first_try)

eta = .01
rew = []
add_scale = 2
print "Training..."
for count in xrange(200):
    print "Iteration #" + str(count)
    params2 = params + add_scale*(np.random.rand(4)-.5)
    r1 = eval(params,env,False)
    r2 = eval(params2,env,False)
    if r2 > r1:
        params = params2
        rew.append(r2)
    else:
        rew.append(r1)

test_try = eval(params,env,True)
print "Test try got to t = " + str(test_try)


plt.plot(rew)
plt.ylabel('Max timestep')
plt.xlabel('Epoch')
plt.show()
