import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, data, epochs=3):
        """
        Args:
            epochs: number of training epochs
        """
        self.epochs = epochs
        self.misclassifications = 0
        self.x_vector = [1, 0, 0] # store all Xi here
        self.weight_vector = [0,1,0.5]
        self.nu = 0.2   # this is the "learning rate" variable
        self.data = data


        ## Add any other variables you need here
        ## YOUR CODE HERE

    def update_weights(self, feature, true_label, i):
        """
        The weight update rule. Iterates over each weight and updates it.
        Increments self.misclassifications by 1 if there is a misclassification.

        Args:
            features: Dependent variables (x)
            true_label: Target variable (y)
        """

        temp_var = (self.weight_vector[0] + (self.weight_vector[1] * feature[0]) + (self.weight_vector[2] * feature[1]) )
        new_weights = self.weight_vector




        if (self.predict(feature) != true_label[i]):
            if(feature[0] * (-1 / self.weight_vector[2]) < feature[1]):
                d = -1
            else:
                d = 1

            new_weights[0] = self.weight_vector[0] + self.nu * d * 1               # d should be 1 if should be in upper
            new_weights[1] = self.weight_vector[1] + self.nu * d * feature[0]      # d should be -1 if should be in lower
            new_weights[2] = self.weight_vector[2] + self.nu * d * feature[1]
            self.misclassifications = self.misclassifications + 1
            self.weight_vector = new_weights
        #
        # if(true_label != self.predict(features)):  # if(misclassified) i might have this backwards
        #     d = -1
        #     self.misclassifications = self.misclassifications + 1
        # else:
        #     d = 1
        #
        # new_omega = []
        # new_omega[0] = self.omega_vector[0] + (self.nu * d * 1) # features[0] needs to = 1
        # new_omega[1] = self.omega_vector[1] + (self.nu * d * features[1])
        # new_omega[1] = self.omega_vector[1] + (self.nu * d * features[2])
        # self.weight_vector = new_omega

    def train(self, features, true_labels, plotting=True):
        """
        features: dependent variables (x)
        true_labels: target variables (y)
        plotting: plot the decision boundary (True by default)
        """
        # Initialize the weights
        self.weight_vector = [0, 1, 0.5]     # right now i think the weights and omega_vector are the same thing
                                                              # might initialize weight vector to [0, 0, 0]
        # For each epoch
        for epoch in range(self.epochs):
            # Iterate over the training data
            for i in range(len(features)):

                if plotting:
                    print("Iteration {}, Misclassifications = {}".
                          format(epoch * len(features) + i + 1, self.misclassifications))
                    self.plot_classifier(features, true_labels, features[i])

                # Update the weights
                self.update_weights(features[i], true_labels, i)    # update the weights with new x,y

            print("=" * 25)
            print("Epoch {}, Accuracy = {}".format(epoch + 1, 1 - self.misclassifications / len(features)))
            print("=" * 25)
            self.misclassifications = 0

    def predict(self, features):    # this is the function that does the dot product? i think?  # this is basically done?
        """                         # features is a 2d array
        Predict the label using self.weights.

        Args:
            features: dependent variables (x)

        Returns:
            The predicted label.
        """
        # return_vector = np.arange(10)
        # for i in range(len(self.data)):
        #     if (self.weight_vector[0] + self.weight_vector[1]*features[i][1] + self.weight_vector[2]*features[i][2]) > 0:
        #         return_vector[i] = 1
        #     else:
        #         return_vector[i] = -1
        if((self.weight_vector[0] + self.weight_vector[1]*features[0] + self.weight_vector[2]*features[1]) > 0 ):
            return 1    # I think this function needs to return an array : make a for loop
        else:
            return -1   # I think this function needs to return an array : make a for loop


    def plot_classifier(self, features, true_labels, data_point):
        """
        Plot the decision boundary.

        Args:
            features: dependent variables (x)
            true_labels: target variables (y)
            data_point: the current data point under consideration
        """
        # Create a mesh to plot
        x1_min, x1_max = features[:, 0].min() - 2, features[:, 0].max() + 2
        x2_min, x2_max = features[:, 1].min() - 2, features[:, 1].max() + 2
        x1x1, x2x2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                                 np.arange(x2_min, x2_max, 0.02))

        Z = np.zeros(x1x1.shape)
        fig, ax = plt.subplots()
        for i in range(len(x1x1)):
            for j in range(len(x1x1[0])):
                temp_var = [x1x1[i, j], x2x2[i, j]]
                Z[i, j] = self.predict([x1x1[i, j], x2x2[i, j]])

        # Put the result into a color plot
        ax.contourf(x1x1, x2x2, Z, cmap='bwr', alpha=0.3)

        # Plot the training points
        plt.scatter(features[:, 0], features[:, 1], c=true_labels, cmap='bwr')
        plt.plot(data_point[0], data_point[1], color='k', marker='x', markersize=12)

        ax.set_title('Perceptron')

        plt.show()


if __name__ == '__main__':

    data = np.array([[2.7810836, 2.550537003, -1],
                     [1.465489372, 2.362125076, -1],
                     [3.396561688, 4.400293529, -1],
                     [1.38807019, 1.850220317, -1],
                     [3.06407232, 3.005305973, -1],
                     [7.627531214, 2.759262235, 1],
                     [5.332441248, 2.088626775, 1],
                     [6.922596716, 1.77106367, 1],
                     [8.675418651, -0.242068655, 1],
                     [7.673756466, 3.508563011, 1]])

    # plt.figure()
    # plt.scatter(data[:, 0], data[:, 1], c=data[:, -1], cmap='bwr')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()

    # features = np.empty([2,len(data)])
    # for i in range(len(data)):
    #     features.insert(i,[[data[i][0]],data[i][1]])

    features = np.array([[data[0][0], data[0][1]], [data[1][0], data[1][1]], [data[2][0], data[2][1]], [data[3][0], data[3][1]], [data[4][0], data[4][1]], [data[5][0], data[5][1]], [data[6][0], data[6][1]], [data[7][0], data[7][1]], [data[8][0], data[8][1]], [data[9][0], data[9][1]]])  # I should automate this process

    true_labels = np.arange(10)

    for i in range(len(data)):
        true_labels[i] = data[i][2]

    percy = Perceptron(data)    # instantiate a perceptron class
    percy.train(features, true_labels)               # train the class on 3 rounds of data



