from sklearn.linear_model import LinearRegression
import numpy as np

def trainModel(df):
    X_train = df.loc[:,  "TV"]
    Y_train = df.loc[:, "Sales"]
    X_train = np.array(X_train).reshape(-1, 1)
    Y_train = np.array(Y_train).reshape(-1, 1)
    model = LinearRegression().fit(X_train, Y_train)
    b = model.intercept_
    w = model.coef_
    return X_train, Y_train, model