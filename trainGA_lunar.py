#Imports
import GA
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
import time

def evaluateIndividual(individual, env, nEvals, visualize, seed=123123123123):
    fitness = 0
    env.seed(seed)
    for i in range(nEvals):
        state = env.reset()
        state = state[None,:]
        finish_episode = False

        while not finish_episode:
            if visualize:
                env.render()

            action = individual.getAction(normalizeState(state))
            new_state, reward, finish_episode, _ = env.step(action)
            state = new_state[None,:]
            fitness += reward

    return fitness/nEvals


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


popSize = 60
networkShape = (8, 50, 50, 4)
init_mu = 0
init_sigma = 0.01

mutateProb = 5/3100
creepRate = 0.001
crossoverProb = 0
pTour = 0.75
tourSize = 8
elitism = 2

nGens = 200
nEvals = 15
successThres = 195

lunarGA = GA.GeneticAlgorithm(populationSize=popSize, evalFunc=evaluateIndividual, networkShape=networkShape, mu=init_mu, sigma=init_sigma)
env = gym.make('LunarLander-v2')

fitnessHist = []
convergedCount = 0
for genIdx in range(nGens):
    lunarGA.nextGeneration(mutateProb=mutateProb, creepRate=creepRate, crossoverProb=crossoverProb,
                          pTour=pTour, tourSize=tourSize, nrElitism=elitism, gymEnv = env, nEvals=nEvals, visualize=False)
    genFitness = lunarGA.getMaxFitness()
    fitnessHist.append(genFitness)
    print("Fitness in gen {} is {}".format(genIdx, fitnessHist[genIdx]))

    # Break if 5 consecutive runs above successThres. Unneccesary if using fixed seed
    if genFitness > successThres:
        convergedCount +=1
    else:
        convergedCount = 0
    if convergedCount > 4:
        break

env.close()
fittest_ind = lunarGA.getFittesetIndividual()

# Save fittest to file
with open('fittest_lunar.pobj', 'wb') as lunar_file:
    pickle.dump(fittest_ind, lunar_file)

 # visualize one run
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

print("Fitness for fittest ind in visualization {}".format(fitness))
env.close()

# validate fittest in training
valFitness = 0
nValidations = 150
for i in range(nValidations):
    state = env.reset()
    state = state[None,:]
    finish_episode = False

    while not finish_episode:
        action = fittest_ind.getAction(normalizeState(state))
        new_state, reward, finish_episode, _ = env.step(action)
        valFitness += reward
        state = new_state[None,:]

valFitness /= nValidations
print("valFitness over {} runs is {}".format(nValidations, valFitness))


fig, ax = plt.subplots()
ax.plot(fitnessHist)
ax.set_xlabel('Generation')
ax.set_ylabel('Fitness')
plt.show()
