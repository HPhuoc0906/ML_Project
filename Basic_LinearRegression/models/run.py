from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("../data/tvmarketing.csv")
    return df.dropna(how = 'all')

def trainModel(df):
    X_train = df.loc[:,  "TV"]
    Y_train = df.loc[:, "Sales"]
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train).reshape(-1, 1)
    model = LinearRegression().fit(X_train, Y_train)
    b = model.intercept_
    w = model.coef_
    return X_train, Y_train, model

def visualize(X_train, Y_train, model):
    plt.scatter(X_train, Y_train)
    plt.xlabel("TV")
    plt.ylabel("Sales")
    x = np.array([min(X_train), max(X_train)])
    y = model.predict(x)
    plt.plot(x, y, color = "r")
    plt.show()

def main():
    df = load_data()
    X_train, Y_train, model = trainModel(df)
    visualize(X_train, Y_train, model)

main()