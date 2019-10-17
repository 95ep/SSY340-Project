#Imports
import GA
import numpy as np
import gym
import matplotlib.pyplot as plt

popSize = 50
networkShape = (4, 50, 50 ,2)
init_mu = 0
init_sigma = 0.01

mutateProb = 5/2900
creepRate = 0.001
crossoverProb = 0
pTour = 0.75
tourSize = 4
elitism = 1

nGens = 150
nEvals = 15
successThres = 390

cartPoleGA = GA.GeneticAlgorithm(populationSize=popSize, networkShape=networkShape, mu=init_mu, sigma=init_sigma)
env = gym.make('CartPole-v0')
env._max_episode_steps = 400

fitnessHist = []
convergedCount = 0
for genIdx in range(nGens):
    cartPoleGA.nextGeneration(mutateProb=mutateProb, creepRate=creepRate, crossoverProb=crossoverProb,
                          pTour=pTour, tourSize=tourSize, nrElitism=elitism, gymEnv = env, nEvals=nEvals, visualize=False)
    genFitness = cartPoleGA.getMaxFitness()
    fitnessHist.append(genFitness)
    print("Fitness in gen {} is {}".format(genIdx, fitnessHist[genIdx]))

    # Break if 5 consecutive runs above successThres
    if genFitness > successThres:
        convergedCount +=1
    else:
        convergedCount = 0
    if convergedCount > 4:
        break

env.close()
fittest_ind = cartPoleGA.getFittesetIndividual()

 # visualize one run
state = env.reset()
state = state[None,:]
finish_episode = False
step = 0
while not finish_episode:
    step += 1
    env.render()
    action = fittest_ind.getAction(state)
    new_state, _, finish_episode, _ = env.step(action)
    state = new_state[None,:]

print("Steps for fittest ind in visualization {}".format(step))
env.close()

# validate fittest in training
valFitness = 0
nValidations = 500
for i in range(nValidations):
    state = env.reset()
    state = state[None,:]
    finish_episode = False

    while not finish_episode:
        action = fittest_ind.getAction(state)
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
