#Imports
import GA
import numpy as np
import gym

popSize = 50
nGenes = 702
init_mu = 0
init_sigma = 0.1

mutateProb = 1/nGenes
creepRate = 0.01
crossoverProb = 1/popSize
pTour = 0.75
tourSize = 2
elitism = 1

nGens = 50

cartPoleGA = GA.GeneticAlgorithm(populationSize=popSize, nGenes=nGenes, mu=init_mu, sigma=init_sigma)
env = gym.make('CartPole-v0')

fitnessHist = np.zeros(nGens)
for genIdx in range(nGens):
    cartPoleGA.nextGeneration(mutateProb=mutateProb, creepRate=creepRate, crossoverProb=crossoverProb,
                          pTour=pTour, tourSize=tourSize, nrElitism=elitism, gymEnv = env, visualize=True)
    fitnessHist[genIdx] = cartPoleGA.getMaxFitness()
    fittest_ind = cartPoleGA.getFittesetIndividual()

    print("Fitness in gen {} is {}".format(genIdx, fitnessHist[genIdx]))

env.close()
