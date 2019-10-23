import numpy as np
import matplotlib.pyplot as plt

cartFile1 = 'results/cartpole_cloud4_training.npy'
cartFile2 = 'results/cartpole_cloud5_training.npy'
cartFile3 = 'results/cartpole_cloud6_training.npy'

lunarFile1 = 'results/lunar_cloud4_training.npy'

# Load training fitness hist and remove trailing zeros (due to early stopping)
cartHist1 = np.trim_zeros(np.load(cartFile1))
cartHist2 = np.trim_zeros(np.load(cartFile2))
cartHist3 = np.trim_zeros(np.load(cartFile3))

lunarHist1 = np.load(lunarFile1)

fig, ax = plt.subplots()
ax.plot(cartHist1, 'b')
ax.plot(cartHist2, 'g')
ax.plot(cartHist3, 'r')
ax.plot(lunarHist1, 'orange')
ax.legend(['CartPole1', 'CartPole2', 'CartPole3', 'Lunar1'], fontsize=14)
ax.set_xlabel('Generation', fontSize=14)
ax.set_ylabel('Fitness', fontSize=14)
plt.show()
