import matplotlib.pyplot as plt

from algorithms.perceptron import Perceptron
from algorithms.adaline_gd import AdalineGD
from algorithms.adaline_sgd import AdalineSGD
from datasets.iris_setosa_dataset import IrisSetosaDataSet
from visualize.decision_boundary_plot import plot_decision_regions

"""Load dataset
-------------------------------------------
"""
iris_setosa_ds = IrisSetosaDataSet()
iris_setosa_ds.read_dataset()
iris_setosa_ds.make_training_data()
iris_setosa_ds.standardize_featues()


"""Create model
-------------------------------------------
"""
nn = AdalineSGD(eta=0.01, n_itr=20)
nn.fit(iris_setosa_ds.X, iris_setosa_ds.y)


"""Plot loss curve
-------------------------------------------
"""
plt.plot(range(1, len(nn.get_errors())+1), 
         nn.get_errors(), marker='o')
plt.xlabel('Epochs')
plt.ylabel("Training Error")
plt.show()


"""Plot decision region
-------------------------------------------
"""
plot_decision_regions(iris_setosa_ds.X, iris_setosa_ds.y, classifier=nn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()