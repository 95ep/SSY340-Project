import numpy as np

class GeneticAlgorithm():
    def __init__(self, populationSize, nGenes, mu, sigma):
        #self.__nGenes = nGenes
        self.__popSize = populationSize
        self.__population = np.random.normal(mu, sigma, (populationSize, nGenes))
        self.__generationIdx = 0
        self.__maxFitness = 0
        self.__fittestIndividual = None
     
    # Getter functions
    def getPopulation(self):
        return self.__population.copy()
    
    
    def getFittesetIndividual(self):
        return self.__fittestIndividual
    
    
    def getMaxFitness(self):
        return self.__maxFitness
    
    def getGenerationIdx(self):
        return self.__generationIdx
    
    
    # Publict methods
    def nextGeneration(self, mutateProb, creepRate, crossoverProb, pTour, tourSize, nrElitism):
        # Increment idx
        self.__generationIdx += 1
        
        fitness = np.zeros(self.__popSize)
        for i in range(self.__popSize):
            individual = self.__population[i,:]
            assert individual.shape == (2,)
            fitness[i] = self.__evaluateIndividual(individual)
            if fitness[i] > self.__maxFitness:
                self.__maxFitness = fitness[i]
                self.__fittestIndividual = individual.copy()
                
        tmpPop = self.__population.copy()
        for i in range(0, self.__popSize, 2):
            i1 = self.__tournamentSelect(fitness, pTour, tourSize)
            i2 = self.__tournamentSelect(fitness, pTour, tourSize)
            individual1 = self.__population[i1, :]
            individual2 = self.__population[i2, :]
            assert individual1.shape == (2,), "Ind 1 is {} with i1 = {}".format(individual1, i1)
            assert individual2.shape == (2,), "Ind 2 is {} with i2 = {}".format(individual2, i2)
            
            r = np.random.rand()
            if r < crossoverProb:
                newChromosomePair = self.__cross(individual1, individual2);
                tmpPop[i1,:] = newChromosomePair[0,:]
                tmpPop[i2,:] = newChromosomePair[1,:]
            else:
                tmpPop[i1,:] = individual1
                tmpPop[i2,:] = individual2
                
        # Mutate
        for i in range(self.__popSize):
            tmpPop[i,:] = self.__creepMutate(tmpPop[i,:], mutateProb, creepRate)
            
        # Elitism
        
        tmpPop = self.__insertFittestIndividual(tmpPop, self.__fittestIndividual, nrElitism)
        
        self.__population = tmpPop.copy()
        assert self.__population.shape == (self.__popSize,2)
                    
    
    # Private helper functions
    def __evaluateIndividual(self, chromosome):
        x = chromosome
        product = 1 + (x[0]-3)**2 + (x[1]-2)**2
    
        f = 1/product;
        return f
    
    
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
        # Single point crossover
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
                
        return newChromosomePair
       
    
    def __insertFittestIndividual(self, pop, fittestIndividual, nInsertions):
        for i in range(nInsertions):
            pop[i,:]  = fittestIndividual
            
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
            
