import numpy as np

class GeneticAlgorithm():
    def __init__(self, populationSize, nGenes, mu, sigma):
        self.__nGenes = nGenes
        self.__populationSize = populationSize
        self.__population = np.random.normal(mu, sigma, (populationSize, nGenes))
        self.__generationIdx = 0
        self.__maxFitness = 0
        self.__fittestIndividualIdx = None
     
    # Getter functions
    def getPopulation(self):
        return self.__population.copy()
    
    
    def getFittesetIndividual(self):
        pass
    
    
    def getGenerationIdx(self):
        pass
    
    
    # Publict methods
    def nextGeneration(self):
        # Increment idx
        self.__generationIdx += 1
        
        fitness = np.zeros(self.__populationSize)
        for i in range(self.__populationSize):
            individual = self.__population[i,:]
            fitness[i] = self.__evaluateIndividual(individual)
            if fitness[i] > self.__maxFitness:
                self.__maxFitness = fitness[i]
                self.__fittestIndividualIdx = i
                
        tmpPop = self.__population.copy()
        for i in range(0, self.__popSize, 2):
            i1 = self.__tournamentSelect(fitness, tournamentSelectionParameter, tournamentSize)
            i2 = self.__tournamentSelect(fitness, tournamentSelectionParameter, tournamentSize)
            individual1 = self.__population[i1, :]
            individual2 = self.__population[i2, :]
            
            r = np.random.rand()
            if r < self.__crossoverProb:
                newChromosomePair = self.__cross(individual1, individual2);
                tmpPop[i1,:] = newChromosomePair[0,:]
                tmpPop[i2,:] = newChromosomePair[1,:]
            else:
                tmpPop[i1,:] = individual1
                tmpPop[i2,:] = individual2
                
        # Mutate
        for i in range(self.__popSize):
            tmpPop[i,:] = self.__mutate(tmpPop[i,:])
            
        # Elitism
        fittestIndividual = tmpPop[self.__fittestIndividualIdx, :]
        tmpPop = self.__insertFittestIndividual(tmpPop, fittestIndividual, nrFittestIndividual)
        
        self.__population = tmpPop.copy()
                    
    
    # Private helper functions
    def __evaluateIndividual(self, chromosome):
        x = self.__decodeChromosome(chromosome)
        factor1 = 1 + (x(1) + x(2) + 1)^2 * (19-14*x(1) + 3*x(1)^2 - 14*x(2) + 6*x(1)*x(2) + 3*x(2));
        factor2 = 30 + (2*x(1) - 3*x(2))^2 * (18-32*x(1) + 12*x(1)^2 + 48*x(2) - 36*x(1)*x(2) + 27*x(2)^2);
        product = factor1 * factor2;
    
        f = 1/product;
        return f
    
    
    def __decodeChromosome(self, chromosome):
        nVariables = 2 # This is hard coded right now
        
        x = np.zeros(nVariables);
        nGenes = chromosome.shape[1]   
        genesPerVariable = nGenes/nVariables;
        assert (nGenes%nVariables == 0)
    
        for n in range(nVariables):
            for j in range(genesPerVariable):
                i = (n-1)*genesPerVariable + j;
                x(n) = x(n) + chromosome(i) * 2^(-j);
        
            x(n) = -variableRange + 2*variableRange*x(n)/(1 - 2^(-genesPerVariable));
    
    
    def __cross(self, chromosome1, chromosome2):
        pass
    
    
    def __evaluateIndividual(self, chromosome):
        pass
    
    
    def __insertBestIndividual(self, population, fittestIndividual, nInsertions):
        pass
    
    
    def __mutate(chromosome, mutationProbability):
        pass
    
    
    def __tournamentSelect(fitness, pTournament, tournamentSize):
        pass
