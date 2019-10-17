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


    def getAction(self, state):
        # Perhaps should force to (0, 1)
        value = np.transpose(state.copy())
        for i in range(len(self.__weights)):
            value = np.matmul(self.__weights[i], value) - self.__biases[i]
            # Add relu
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

        '''# To be improved
        nGenes = chromosome.shape[0]
        mutatedChromosome = chromosome.copy()
        for i in range(nGenes):
            r = np.random.rand()
            if (r < mutationProb):
                mutatedChromosome[i] += np.random.normal(0, creepRate)

        return mutatedChromosome'''


class GeneticAlgorithm():
    def __init__(self, populationSize, networkShape, mu, sigma):
        #self.__nGenes = nGenes
        self.__popSize = populationSize
        #self.__networkShape
        #prevLayerDim = 1
        #nGenes = 0
        #for layerDim in networkShape:
            #nGenes += layerDim

        #self.__population = np.random.normal(mu, sigma, (populationSize, nGenes))
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
    def nextGeneration(self, mutateProb, creepRate, crossoverProb, pTour, tourSize, nrElitism, gymEnv, visualize=False):
        # Increment idx
        self.__generationIdx += 1

        fitness = np.zeros(self.__popSize)
        for i in range(self.__popSize):
            individual = self.__population[i]
            #assert individual.shape == (self.__nGenes,)
            fitness[i] = self.__evaluateIndividual(individual, gymEnv, visualize)
            if fitness[i] > self.__maxFitness:
                self.__maxFitness = fitness[i]
                self.__fittestIndividual = copy.deepcopy(individual)

        tmpPop = self.__population.copy()
        for i in range(0, self.__popSize, 2):
            i1 = self.__tournamentSelect(fitness, pTour, tourSize)
            i2 = self.__tournamentSelect(fitness, pTour, tourSize)
            individual1 = copy.deepcopy(self.__population[i1])
            individual2 = copy.deepcopy(self.__population[i2])
            #assert individual1.shape == (self.__nGenes,)
            #assert individual2.shape == (self.__nGenes,)

            r = np.random.rand()
            # Cross not implemented right now
            if r < crossoverProb:
                newChromosomePair = self.__cross(individual1, individual2);
                tmpPop[i1] = newChromosomePair[0]
                tmpPop[i2] = newChromosomePair[1]
            else:
                tmpPop[i1] = individual1
                tmpPop[i2] = individual2

        # Mutate
        for i in range(self.__popSize):
            tmpPop[i].creepMutate(mutateProb, creepRate)

        # Elitism
        tmpPop = self.__insertFittestIndividual(tmpPop, self.__fittestIndividual, nrElitism)

        self.__population = copy.deepcopy(tmpPop)
        #assert self.__population.shape == (self.__popSize, self.__nGenes)


    # Private helper functions
    def __evaluateIndividual(self, individual, env, visualize):

        state = env.reset()
        state = state[None,:]
        finish_episode = False
        fitness = 0
        while not finish_episode:
            if visualize:
                env.render()
            action = individual.getAction(state)
            new_state, reward, finish_episode, _ = env.step(action)
            state = new_state[None,:]
            fitness += reward

        return fitness


    '''def __getAction(self, chromosome, state):
        inputLayer = np.ndarray((4,1), dtype=float, buffer=state)
        assert inputLayer.shape == (4,1)
        weights1 = np.ndarray((100,4), dtype=float, buffer = chromosome[0:400])
        assert weights1.shape == (100,4)
        bias1 = np.ndarray((100,1), dtype=float, buffer = chromosome[400:500])
        assert bias1.shape == (100,1)
        weights2 = np.ndarray((2, 100), dtype=float, buffer=chromosome[500:700])
        assert weights2.shape == (2,100)
        bias2 = np.ndarray((2,1), dtype=float, buffer=chromosome[700:702])
        assert bias2.shape == (2,1)

        hidden1 = np.matmul(weights1, inputLayer) - bias1
        # Add relu
        hidden1 = hidden1 * (hidden1 > 0)
        assert hidden1.shape == (100,1)

        hidden2 = np.matmul(weights2, hidden1) - bias2
        assert hidden2.shape == (2, 1)

        action = np.argmax(hidden2)

        return action'''

    '''def __decodeChromosome(self, chromosome):
        nVariables = 2 # This is hard coded right now

        x = np.zeros(nVariables);
        nGenes = chromosome.shape[1]
        genesPerVariable = nGenes/nVariables;
        assert (nGenes%nVariables == 0)

        for n in range(nVariables):
            for j in range(genesPerVariable):
                i = (n-1)*genesPerVariable + j;
                x[n] = x[n] + chromosome(i) * 2^(-j);

            x[n] = -variableRange + 2*variableRange*x[n]/(1 - 2^(-genesPerVariable));

        return x'''


    def __cross(self, chromosome1, chromosome2):
        '''# Single point crossover
        nGenes = chromosome1.shape[0]
        crossoverPoint = np.random.randint(1, nGenes)

        newChromosomePair = np.zeros((2, nGenes))
        for i in range(nGenes):
            if i < crossoverPoint:
                newChromosomePair[0,i] = chromosome1[i]
                newChromosomePair[1,i] = chromosome2[i]
            else:
                newChromosomePair[0,i] = chromosome2[i]
                newChromosomePair[1,i] = chromosome1[i]

        return newChromosomePair'''
        pass


    def __insertFittestIndividual(self, pop, fittestIndividual, nInsertions):
        for i in range(nInsertions):
            pop[i]  = copy.deepcopy(fittestIndividual)

        return pop


    def __creepMutate(self, chromosome, mutationProb, creepRate):
        # To be improved
        nGenes = chromosome.shape[0]
        mutatedChromosome = chromosome.copy()
        for i in range(nGenes):
            r = np.random.rand()
            if (r < mutationProb):
                mutatedChromosome[i] += np.random.normal(0, creepRate)

        return mutatedChromosome


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
        assert idxSelected < self.__popSize, "idxSelected is {}".format(idxSelected)
        return idxSelected
