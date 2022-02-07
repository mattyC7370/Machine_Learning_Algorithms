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
    plt.show()

    # convert the dataframe `df` to a Numpy array so that it is easier to perform operations on it
    X_train = np.array(df['Height'])
    y_train = np.array((df['Weight'] >= 60).astype('float'))
    X_train = np.expand_dims(X_train, -1)

    plt.scatter(df['Height'], y_train, marker='X')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    x = np.linspace(1.4, 1.9, 100)
    plt.plot(x, (1/(1+np.exp((-50)*(x-1.65)))))
    plt.show()


    # Testing - From easy example
    # Weight_Array = np.array([0.5, 2.3, 2.9])
    # Height_Array = np.array([1.4, 1.9, 3.2])

    # Build the model and train on the dataset
    model = LogisticRegression(0.1, 100000)
    model.train(X_train, y_train)



class LogisticRegression:
    def __init__(self, lr=0.001, epochs=30):
        """
                Fits a linear regression model on a given dataset.

                Args:
                    lr: learning rate
                    epochs: number of iterations over the dataset
        """
        self.lr = lr
        self.epochs = epochs
        self.intercept = -3
        self.slope = (3/1.65)
        self.interceptStepSize = 0
        self.slopeStepSize = 0
        self.derivative_of_SSR_Intercept = 100000
        self.derivative_of_SSR_Slope = 100000
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
        self.intercept = -3
        self.slope = (3/1.65)


        # The following while loop will be changed to be dependent on the # of epochs, not this magic number 0.1 (At least I think)
        # -Actually its initializing the LR class with 100000 epochs so it might have to factor in this number
        #while self.derivative_of_SSR_Intercept > 0.1:
        for i in range(self.epochs):    # have some sort of limitation on the epochs
            self.update_weights(X, y)
            self.intercept = self.intercept - self.interceptStepSize
            self.slope = self.slope - self.slopeStepSize


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

        DerivativeSumOfSquaredResidual_Intercept = 0
        DerivativeSumOfSquaredResidual_Slope = 0

        for i in range(len(X)):
            DerivativeSumOfSquaredResidual_Intercept = DerivativeSumOfSquaredResidual_Intercept + (-2 * (y[i] - self.intercept -X[i]*self.slope))
            DerivativeSumOfSquaredResidual_Slope = DerivativeSumOfSquaredResidual_Slope + ((-2 * X[i]) * (y[i] - self.intercept - X[i]*self.slope))

        self.derivative_of_SSR_Intercept = DerivativeSumOfSquaredResidual_Intercept
        self.interceptStepSize = DerivativeSumOfSquaredResidual_Intercept * self.lr

        self.derivative_of_SSR_Slope = DerivativeSumOfSquaredResidual_Slope
        self.slopeStepSize = DerivativeSumOfSquaredResidual_Slope * self.lr


        print("derivative of the SSR at intercept =", self.intercept, " :", DerivativeSumOfSquaredResidual_Intercept)
        print("derivative of the SSR with respect to slope______ at Slope =", self.slope, " :", DerivativeSumOfSquaredResidual_Slope)
        print(self.interceptStepSize, self.slopeStepSize)

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