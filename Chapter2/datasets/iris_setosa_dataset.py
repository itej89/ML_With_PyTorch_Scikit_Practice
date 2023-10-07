import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class IrisSetosaDataSet:

    web_csv_link = 'https://archive.ics.uci.edu/ml/'\
        'machine-learning-databases/iris/iris.data'


    def read_dataset(self):
        print("Loading data set--------")

        self.data_set = pd.read_csv(self.web_csv_link, header=None, encoding='utf-8')
        
        print("Loaded data set--------")
        print(self.data_set.tail())
        print("--------")

    def make_training_data(self):
        self.X = self.data_set.iloc[0:100, [0, 2]].values
        self.y = self.data_set.iloc[0:100, 4].values

        """Create target classes from string data
        """
        self.y = np.where(self.y == "Iris-setosa", 0, 1)

    def plot_training_data(self):
        plt.scatter(self.X[:50, 0], self.X[:50, 1],
                     color='red', marker='s', label='Setosa')
        plt.scatter(self.X[50:100, 0], self.X[50:100, 1],
                     color='blue', marker='s', label='Versicolor')
        plt.xlabel('Sepal length [cm]')
        plt.ylabel('Petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()

    def standardize_featues(self):
        X_std = np.copy(self.X)
        X_std[:, 0] = (self.X[:,0] - self.X[:,0].mean()) / self.X[:,0].std()
        X_std[:, 1] = (self.X[:,1] - self.X[:,1].mean()) / self.X[:,1].std()
        self.X = X_std

if __name__ == "__main__":

    _irisSetosaDataSet = IrisSetosaDataSet()

    _irisSetosaDataSet.read_dataset()
    _irisSetosaDataSet.make_training_data()
    _irisSetosaDataSet.plot_training_data()

