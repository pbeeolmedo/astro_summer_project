import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(color_codes=True)


x = np.random.normal(size=100)
sns.distplot(x)

plt.show()
