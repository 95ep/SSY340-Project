import numpy as np
import matplotlib.pyplot as plt

cartFile1 = 'results/cartpole_cloud4_training.npy'
cartFile2 = 'results/cartpole_cloud5_training.npy'
cartFile3 = 'results/cartpole_cloud6_training.npy'

lunarFile1 = 'results/lunar_cloud4_training.npy'
lunarFile2 = 'results/lunar_cloud5_training.npy'
lunarFile3 = 'results/lunar_cloud6_training.npy'

# Load training fitness hist and remove trailing zeros (due to early stopping)
cartHist1 = np.trim_zeros(np.load(cartFile1))
cartHist2 = np.trim_zeros(np.load(cartFile2))
cartHist3 = np.trim_zeros(np.load(cartFile3))

lunarHist1 = np.load(lunarFile1)
lunarHist2 = np.load(lunarFile2)
lunarHist3 = np.load(lunarFile3)

fig, ax = plt.subplots()
ax.plot(cartHist1, 'b')
ax.plot(cartHist2, 'g')
ax.plot(cartHist3, 'r')
ax.plot(lunarHist1, 'orange', linestyle='solid')
ax.plot(lunarHist2, 'm', linestyle='solid')
ax.plot(lunarHist3, 'y', linestyle='solid')
ax.legend(['CP1', 'CP2', 'CP3', 'L1', 'L2', 'L3'], fontsize=12, loc=(0.8, 0.5))
ax.set_xlabel('Generation', fontSize=12)
ax.set_ylabel('Fitness', fontSize=12)
ax.set_ylim(-210, 210)
ax.set_title('GA')
plt.show()
