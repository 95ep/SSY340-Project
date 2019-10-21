import GA
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle


def normalizeState(state):
    # Normalize the state to work better in the network
    state = np.transpose(state)
    # Somewhat arbitrary normalization based on observed values
    state[0] = state[0]/10
    state[1] = state[1]/20
    state[2] = state[2]/5
    state[3] = state[3]/10
    state[4] = state[4]/20
    state[5] = state[5]/20
    # Clip to ensure that we are within -1 to 1
    #state = np.clip(state, -1, 1)

    return state


with open('fittest_lunar5.pobj', 'rb') as fittest_file:
     fittest_ind = pickle.load(fittest_file)


env = gym.make('LunarLander-v2')
env.seed(int(0))
for i in range(15):
    state = env.reset()
    state = state[None,:]
    finish_episode = False
    fitness = 0
    while not finish_episode:
        env.render()
        action = fittest_ind.getAction(normalizeState(state))
        new_state, reward, finish_episode, _ = env.step(action)
        state = new_state[None,:]
        fitness += reward
    print("Fitness for episode {} is {}".format(i, fitness))

env.close()
