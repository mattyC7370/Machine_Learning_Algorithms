import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("Hello world")
    df = pd.read_csv('./heights_weights.csv')
    df.head()

    import matplotlib.pyplot as plt
    plt.scatter(df['Height'], df['Weight'], marker='X')
    plt.xlabel("Height")
    plt.ylabel("Weight")

    # convert the dataframe `df` to a Numpy array so that it is easier to perform operations on it
    X_train = np.array(df['Height'])
    y_train = np.array(df['Weight'])
    X_train = np.expand_dims(X_train, -1)

    # Build the model and train on the dataset
    model = LinearRegression(0.01, 100000)

    # x = np.linspace(1.4, 1.9, 100)
    # plt.plot(x, (model.intercept + model.slope * x))
    plt.show()

    # is 59 really the best intercept for slope = 2  YES!!!
    model.train(X_train, y_train)



class LinearRegression:
    def __init__(self, lr=0.001, epochs=30):
        """
                Fits a linear regression model on a given dataset.

                Args:
                    lr: learning rate
                    epochs: number of iterations over the dataset
        """
        self.lr = lr
        self.epochs = epochs
        self.intercept = 0
        self.slope = 1
        self.stepSize = 0
        self.derivative_of_SSR = 100000
        ######################
        #   YOUR CODE HERE   #
        ######################
        # You may add additional fields

    def train(self, X, y):
        """
        Initialize weights. Iterate through the dataset and update weights once every epoch.

        Args:
            X: features
            y: target
        """
        ######################
        #   YOUR CODE HERE   #
        ######################
        self.intercept = 0
        self.slope = 1

        while self.derivative_of_SSR > 0.1:
            self.update_weights(X, y)
            self.intercept = self.intercept + self.stepSize



    def update_weights(self, X, y):
        """
        Helper function to calculate the gradients and update weights using batch gradient descent.

        Args:
            X: features
            y: target
        """
        ######################
        #   YOUR CODE HERE   #
        ######################
        # sumOfSquaredResidual = 0
        # for i in range(len(X)):
        #     residual = y[i] - self.predict(X[i])
        #     sumOfSquaredResidual = sumOfSquaredResidual + residual**2
        #
        # print("sumOfSquaredResidual: ", sumOfSquaredResidual)

        runningTotal = 0
        for i in range(len(X)):
            runningTotal = runningTotal + (-2 * (y[i] - self.predict(X[i])))

        self.derivative_of_SSR = abs(runningTotal)
        self.stepSize = abs(runningTotal) * self.lr
        print("derivative of the SSR at intercept =", self.intercept, " :", runningTotal)


    def predict(self, feature):
        """
        Predict values using the weights.

        Args:
            feature: single feature value

        Returns:
            The predicted value.
            :param feature:
        """
        ######################
        #   YOUR CODE HERE   #
        ######################
        predictedWeight = self.intercept + (self.slope * feature)
        return predictedWeight

if __name__ == '__main__':
    main()