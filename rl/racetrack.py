# Racetrack exercise (5.8) from Rich Sutton's Reiforcement Learning book
# Used an Off-Policy Monte Carlo method
#
# Author: Murilo V. F. Menezes

import numpy as np

def eps_greedy(epsilon,Q):


def generate_episode():


def main():
    f = open('map','r')
    track = np.array([map(int,line.split(',')) for line in f])
    f.close()

    n_states = sum(sum(track==1)) + sum(sum(track==3))
    n_actions = 6

    # actions:
    #   0 - increase 0 on vertical axis
    #   1 - increase 1 on vertical axis
    #   2 - increase -1 on vertical axis
    #   3 - increase 0 on horizontal axis
    #   4 - increase 1 on horizontal axis
    #   5 - increase -1 on horizontal axis

    print "Racetrack: "
    print track

    # initialization
    Q = np.random.rand(n_states,n_actions)
    cumulative_weights = np.zeros([n_states,n_actions])

    epsilon = .1

    returns = generate_episode(epsilon)


if __name__ == '__main__':
    main()
