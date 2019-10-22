import numpy as np
import matplotlib.pyplot as plt

fileName1 = 'lunar_test.npy'
fileName2 = 'cartpole_test.npy'

fitnessHist1 = np.load(fileName1)
fitnessHist2 = np.load(fileName2)

fig, ax = plt.subplots()
ax.plot(fitnessHist1)
ax.plot(fitnessHist2)
ax.legend(['Lunar', 'CartPole'], fontsize=14)
ax.set_xlabel('Generation', fontSize=14)
ax.set_ylabel('Fitness', fontSize=14)
plt.show()
