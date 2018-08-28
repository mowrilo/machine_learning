# implementation of the stationary k-armed bandit problem
# Murilo Menezes, 2017
# Aug/18 -> adapted from Python2.7 to Python3.6

import numpy as np
import matplotlib.pyplot as plt
import sys

class bandit:
    def __init__(self,mean,variance):
        self.mu = mean
        self.sigma = variance
    def give_reward(self):
        return np.random.normal(self.mu,self.sigma)


def main():
#generate bandits
    n_bandits = int(sys.argv[1])
    bandit_list = []
    best_bandit = 0
    np.random.seed(2)
    for i in range(n_bandits):
        this_mu = np.random.rand()*10 - 5
        this_var = np.random.rand()+.5
        bandit_list.append(bandit(this_mu,this_var))
        if this_mu > bandit_list[best_bandit].mu:
            best_bandit = i

    total_reward = 0#[0]

#initialize parameters
    Q = [0]*n_bandits
    N = [0]*n_bandits
    epsilon_discount = float(sys.argv[4])
    eps = float(sys.argv[3])
    n_repeats = int(sys.argv[2])
    for count in range(n_repeats):
        if (np.random.rand() > eps):
            A = Q.index(max(Q))
        else:
            A = np.random.randint(n_bandits)
        reward = bandit_list[A].give_reward()
        #total_reward.append((reward+total_reward[count])/((count+1) * bandit_list[best_bandit].mu))
        total_reward += reward
        N[A] = N[A] + 1
        Q[A] = Q[A] + 1/(N[A]) * (reward - Q[A])
        eps = eps*epsilon_discount
    for i in range(n_bandits):
        print("Estimated reward for bandit",str(i),":\t",str(Q[i]))
        print("Real reward for bandit",str(i),":\t",str(bandit_list[i].mu))
        print("Error for bandit",str(i),":\t",str(bandit_list[i].mu - Q[i]))
        print("Variance of bandit",str(i),":\t",str(bandit_list[i].sigma),"\n\n")

    print("Total reward:",total_reward)
    print("Average reward per step:",total_reward/n_repeats)

    #plt.plot(total_reward)
    #plt.show()

if __name__ == "__main__":
    if (sys.argv[1] == "--help"):
        print("Usage: python3 bandits.py p1 p2 p3 p4")
        print("Parameters:")
        print("\tp1: number of arms on the bandit")
        print("\tp2: number of iterations")
        print("\tp3: initial probability of exploration")
        print("\tp4: discount factor on the exploration probability (p3)")
    else:
        if (len(sys.argv) < 5):
            print("You should specify 4 parameters!\nFor help, use \'python3 bandits.py --help\'")
        else:
            main()
