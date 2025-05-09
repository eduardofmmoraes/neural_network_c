from nnfs.datasets import sine_data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

X_train, y_train = sine_data()
X_test, y_test = sine_data()

pd.DataFrame(0.1 * np.random.randn(1, 64)).to_csv("./datasets/sine/weights1.csv", header=False, index=False)
pd.DataFrame(0.1 * np.random.randn(64, 64)).to_csv("./datasets/sine/weights2.csv", header=False, index=False)
pd.DataFrame(0.1 * np.random.randn(64, 1)).to_csv("./datasets/sine/weights3.csv", header=False, index=False)