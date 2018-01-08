# implementation of the stationary k-armed bandit problem
# Murilo Menezes, 2017

import numpy as np
import matplotlib.pyplot as plt

class bandit:
    def __init__(self,mean,variance):
        self.mu = mean
        self.sigma = variance
    def give_reward(self):
        return np.random.normal(self.mu,self.sigma)


def main():
#generate bandits
    n_bandits = 10
    bandit_list = []
    best_bandit = 0
    for i in xrange(n_bandits):
        this_mu = np.random.rand()*10 - 5
        this_var = np.random.rand()+.5
        bandit_list.append(bandit(this_mu,this_var))
        if this_mu > bandit_list[best_bandit].mu:
            best_bandit = i

    total_reward = [0]

#initialize parameters
    Q = [0]*n_bandits
    N = [0]*n_bandits
    eps = .05
    n_repeats = 100000
    for count in xrange(n_repeats):
        if (np.random.rand() > eps):
            A = Q.index(max(Q))
        else:
            A = np.random.randint(n_bandits)
        reward = bandit_list[A].give_reward()
        total_reward.append((reward+total_reward[count])/((count+1) * bandit_list[best_bandit].mu))
        N[A] = N[A] + 1
        Q[A] = Q[A] + 1/(N[A]) * (reward - Q[A])
    for i in xrange(n_bandits):
        print "Estimated reward for bandit " + str(i) + ":\t" + str(Q[i])
        print "Real reward for bandit " + str(i) + ":\t" + str(bandit_list[i].mu)
        print "Error for bandit " + str(i) + ":\t" + str(bandit_list[i].mu - Q[i])
        print "Variance of bandit " + str(i) + ":\t" + str(bandit_list[i].sigma) + "\n\n"

    plt.plot(total_reward)
    plt.show()

if __name__ == "__main__":
    main()
