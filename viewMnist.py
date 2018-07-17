from loadDataset import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
plt.imshow(X_train[0][0] ,cmap=cm.binary)