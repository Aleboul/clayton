from clayton.rng.evd import HuslerReiss
import numpy as np
import matplotlib.pyplot as plt

sigmat = np.array([[0.0,0.5],[0.5,0.0]])

hr1 = HuslerReiss(sigmat=sigmat, n_sample =1000)
sample=hr1.sample_unimargin()

fig, ax = plt.subplots()

ax.scatter(sample[:,0], sample[:,1])

plt.show()