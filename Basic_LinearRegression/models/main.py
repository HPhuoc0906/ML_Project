import visual
from utils import *
import train


if __name__ == "__main__":
    df = load_data("../data/tvmarketing.csv")
    X_train, Y_train, models = train.trainModel(df)
    visual.visualize(X_train, Y_train, models)