import gym
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def mlp_net(data,weights):
    for layer in xrange(len(weights)):
        #print layer
        data = np.tanh(np.dot(weights[layer],data)) 
    exp_data = np.exp(data)
    probs = exp_data/sum(exp_data)
    cumsum = np.cumsum(probs)
    rnd = np.random.rand()
    for n in xrange(len(cumsum)):
        if cumsum[n] > rnd:
            break
    return n

def eval_episode(weights,env,render):
    obs = env.reset()
    total_reward = 0
    while True:#for ts in xrange(500):
        if render:
            env.render()
        #print "tudo ok"
        action = mlp_net(obs,weights)
        obs,reward,fell,info = env.step(action)
        total_reward += reward
        if fell:
            print "Died!"
#            print info
            break
    #print ts
    return total_reward

#def array_to_weights(arr):


env = gym.make('SpaceInvaders-ram-v0')
#params = #np.random.rand(128) - .5
n_outputs = env.action_space.n
weights = []
net_arch = [128,60,30,n_outputs]
total_weights = 128*60 + 60*30 + 30*n_outputs
for i in xrange(len(net_arch)-1):
    w = np.random.rand(net_arch[i+1],net_arch[i])
    weights.append(w)
first_try = eval_episode(weights,env,True)
print "First try got reward = " + str(first_try)

eta = .01
rew = []
add_scale = 2
print "Training..."
for count in xrange(200):
    print "Iteration #" + str(count)
    weights2 = weights
    for i in xrange(len(weights)):
        weights2[i] = weights2[i] + add_scale*(np.random.rand(weights2[i].shape[0],weights2[i].shape[1])-.5)
    #weights2 = (params2 - min(params2))/(max(params2)-min(params2))
    r1 = eval_episode(weights,env,False)
    r2 = eval_episode(weights2,env,False)
    if r2 > r1:
        weights = weights2
        rew.append(r2)
    else:
        rew.append(r1)

test_try = eval_episode(weights,env,True)
print "Test try got reward = " + str(test_try)


plt.plot(rew)
plt.ylabel('Max timestep')
plt.xlabel('Epoch')
plt.show()
