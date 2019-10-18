import numpy as np
import copy

class GAActionNetwork():
    def __init__(self, networkShape, mu, sigma):
        self.__nHiddenLayers = len(networkShape) - 2
        assert self.__nHiddenLayers >= 0, "Invalid networkShape"

        prevLayerDim = networkShape[0]
        weigths = []
        biases = []
        for i in range(1, len(networkShape)):
            layerDim = networkShape[i]
            weigths.append(np.random.normal(mu, sigma, (layerDim, prevLayerDim)))
            biases.append(np.zeros((layerDim, 1)))
            prevLayerDim = layerDim
        assert len(weigths) == len(networkShape)-1

        self.__weights = weigths
        self.__biases = biases


    def getAction(self, input_state):
        # Forward prob through network
        value = input_state.copy()
        for i in range(len(self.__weights)):
            value = np.matmul(self.__weights[i], value) - self.__biases[i]
            # Apply relu
            if (i < len(self.__weights) - 1):
                value = value * (value > 0)

        action = np.argmax(value)
        return action


    def creepMutate(self, mutationProb, creepRate):
        # I doubt that this actually works
        for weightMatrix in self.__weights:
            for w in weightMatrix:
                r = np.random.rand()
                if (r < mutationProb):
                    w += np.random.normal(0, creepRate)

        for biasVec in self.__biases:
            for b in biasVec:
                r = np.random.rand()
                if (r < mutationProb):
                    b += np.random.normal(0, creepRate)


class GeneticAlgorithm():
    def __init__(self, populationSize, evalFunc, networkShape, mu, sigma):
        self.__popSize = populationSize
        self.__evaluateIndividual = evalFunc

        pop = []
        for i in range(populationSize):
            pop.append(GAActionNetwork(networkShape, mu, sigma))
        self.__population = pop
        self.__generationIdx = 0
        self.__maxFitness = 0
        self.__fittestIndividual = None

    # Getter functions
    def getPopulation(self):
        return copy.deepcopy(self.__population)


    def getFittesetIndividual(self):
        return self.__fittestIndividual


    def getMaxFitness(self):
        return self.__maxFitness

    def getGenerationIdx(self):
        return self.__generationIdx


    # Publict methods
    def nextGeneration(self, mutateProb, creepRate, crossoverProb, pTour, tourSize, nrElitism, gymEnv, nEvals, visualize=False):
        # Increment idx
        self.__generationIdx += 1
        self.__maxFitness = 0

        fitness = np.zeros(self.__popSize)
        for i in range(self.__popSize):
            individual = self.__population[i]
            fitness[i] = self.__evaluateIndividual(individual, gymEnv, nEvals, visualize, self.__generationIdx)

            if fitness[i] > self.__maxFitness:
                self.__maxFitness = fitness[i]
                self.__fittestIndividual = copy.deepcopy(individual)

        tmpPop = self.__population.copy()
        for i in range(0, self.__popSize, 2):
            i1 = self.__tournamentSelect(fitness, pTour, tourSize)
            i2 = self.__tournamentSelect(fitness, pTour, tourSize)
            #print("i1 = {} and i2 = {}".format(i1, i2))

            individual1 = copy.deepcopy(self.__population[i1])
            individual2 = copy.deepcopy(self.__population[i2])
            #assert individual1.shape == (self.__nGenes,)
            #assert individual2.shape == (self.__nGenes,)
            tmpPop[i] = individual1
            tmpPop[i+1] = individual2

        # Mutate
        for i in range(self.__popSize):
            tmpPop[i].creepMutate(mutateProb, creepRate)

        # Elitism
        tmpPop = self.__insertFittestIndividual(tmpPop, self.__fittestIndividual, nrElitism)

        self.__population = copy.deepcopy(tmpPop)
        #assert self.__population.shape == (self.__popSize, self.__nGenes)


    # Private helper functions



    def __insertFittestIndividual(self, pop, fittestIndividual, nInsertions):
        for i in range(nInsertions):
            pop[i]  = copy.deepcopy(fittestIndividual)

        return pop


    '''def __creepMutate(self, chromosome, mutationProb, creepRate):
        # To be improved
        nGenes = chromosome.shape[0]
        mutatedChromosome = chromosome.copy()
        for i in range(nGenes):
            r = np.random.rand()
            if (r < mutationProb):
                mutatedChromosome[i] += np.random.normal(0, creepRate)

        return mutatedChromosome'''


    def __tournamentSelect(self, fitness, pTournament, tournamentSize):
        # Index of chromosomes in tournament
        inTournament = np.random.randint(0, self.__popSize, tournamentSize)

        while (tournamentSize > 1):
            r = np.random.rand()
            fittestInTournament = np.argmax(fitness[inTournament])

            if (r < pTournament): # Select fittest in tournament
                idxSelected = inTournament[fittestInTournament]
                break
            else: # Remove the fittest chromosome from tournament
                inTournament = np.concatenate((inTournament[0:fittestInTournament], inTournament[fittestInTournament+1:]))
                tournamentSize -= 1

            # Select remaining chromosome when only one left
            if (tournamentSize == 1):
                idxSelected = inTournament[0]
        #assert idxSelected < self.__popSize, "idxSelected is {}".format(idxSelected)
        #print("Fitness of selected {}".format(fitness[idxSelected]))
        return idxSelected
