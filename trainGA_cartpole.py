#Imports
import GA
import numpy as np
import gym
import matplotlib.pyplot as plt

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
    state[0] = (state[0]+4.8)/9.6
    state[2] = (state[2]+24*2*np.pi/360)/(48*2*np.pi/360)
    return state


popSize = 60
networkShape = (4, 64, 64 ,2)
init_mu = 0
init_sigma = 0.01

mutateProb = 5/4610
creepRate = 0.001
crossoverProb = 0
pTour = 0.75
tourSize = 4
elitism = 2

nGens = 150
nEvals = 15
successThres = 195

cartPoleGA = GA.GeneticAlgorithm(populationSize=popSize, evalFunc=evaluateIndividual, networkShape=networkShape, mu=init_mu, sigma=init_sigma)
env = gym.make('CartPole-v0')
env._max_episode_steps = 200

fitnessHist = []
convergedCount = 0
for genIdx in range(nGens):
    cartPoleGA.nextGeneration(mutateProb=mutateProb, creepRate=creepRate, crossoverProb=crossoverProb,
                          pTour=pTour, tourSize=tourSize, nrElitism=elitism, gymEnv = env, nEvals=nEvals, visualize=False)
    genFitness = cartPoleGA.getMaxFitness()
    fitnessHist.append(genFitness)
    print("Fitness in gen {} is {}".format(genIdx, fitnessHist[genIdx]))

    # Break if 5 consecutive runs above successThres. Unneccesary if using fixed seed
    if genFitness > successThres:
        convergedCount +=1
    else:
        convergedCount = 0
    if convergedCount > 6:
        break

env.close()
fittest_ind = cartPoleGA.getFittesetIndividual()

# Save fittest to file
with open('fittest_cartpole.pobj', 'wb') as lunar_file:
    pickle.dump(fittest_ind, lunar_file)

# validate fittest in training
valFitness = 0
nValidations = 500
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
plt.savefig('cartpole.png')
