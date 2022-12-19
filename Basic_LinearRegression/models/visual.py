import matplotlib.pyplot as plt
import numpy as np

def visualize(X_train, Y_train, model):
    plt.scatter(X_train, Y_train)
    plt.xlabel("TV")
    plt.ylabel("Sales")
    x = np.array([min(X_train), max(X_train)])
    y = model.predict(x)
    plt.plot(x, y, color = "r")
    plt.show()
