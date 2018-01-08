# Monte Carlo policy estimation method
#       to play (a simplified version of) Blackjack
# The "dealer" is just a random variable
# In this first implementation, there is no Ace,
#       just cards from 2 to 10, with reposition
#
#
# Author: Murilo V. F. Menezes

import numpy as np

def hit():
    return np.random.randint(2,11)

def dealer(sum_player):
    sum_dealer = np.random.randint(12,17) + hit()#.poisson(3,1)[0] + 14
    sum_dealer = int(sum_dealer)
    if sum_dealer < sum_player and sum_player <= 21:
        sum_dealer += hit()
    return sum_dealer

def eval_game(sum_player,sum_dealer):
    reward = 0

    if sum_player > 21:
        if sum_dealer <= 21:
            reward = -1
    else:
        if sum_dealer < sum_player:
            reward = 1
        elif sum_dealer > sum_player:
            if sum_dealer > 21:
                reward = 1
            else:
                reward = -1

    return reward

# Action 0: hit
# Action 1: stick
def sim_episode(action_values,verbose=False):
    sum_player = hit() + hit()
    rew = 0
    last_state = sum_player
    while True:
        last_state = sum_player
        if action_values[sum_player-2][0] == action_values[sum_player-2][1]:
            action = np.random.randint(0,2) # ties are broken randomly
        else:
            action = action_values[sum_player-2].argmax()
        if verbose:
            print "Sum of player: " + str(sum_player) + "\tAction: " + str(action)
        if action == 0:
            sum_player += hit()
        else:
            break

        if sum_player > 21:
            break

    sum_dealer = dealer(sum_player)
    rew = eval_game(sum_player,sum_dealer)

    if verbose:
        print "\nFinal sum of player: " + str(sum_player) + "\nFinal sum of dealer: " + str(sum_dealer) + "\nReward: " + str(rew)

    return (rew,last_state,action)


def main():
    # states: sum of cards from 2 to 21
    action_values = np.zeros([20,2])
    times_visited = np.zeros([20,2])
    n_episodes = 1000000
    print "Training the agent in " + str(n_episodes) + " episodes..."
    for i in xrange(n_episodes):
        rew,last_state,action = sim_episode(action_values)
        action_values[last_state-2][action] *= times_visited[last_state-2][action]
        action_values[last_state-2][action] += rew
        times_visited[last_state-2][action] += 1
        action_values[last_state-2][action] /= times_visited[last_state-2][action]
    print "Done!\n"


    print "Test game:"
    rew,last_state,action = sim_episode(action_values,verbose=True)
    print "Last state: " + str(last_state) + "\tAction: " + str(action)

    print "Final action values:\n"
    print action_values


if __name__ == "__main__":
    main()
