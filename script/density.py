from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('alpha.csv')
X=df['alpha'].to_numpy()
X=X.reshape(-1, 1)
print(X.shape)
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)

t = np.arange(0.0, 1.0, 0.01).reshape(-1, 1)

y=kde.score_samples(t)
fig, ax = plt.subplots()

ax.plot(t, y)
plt.show()