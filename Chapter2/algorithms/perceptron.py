import numpy as np

class Perceptron:
    """Perceptron Classifier
    """


    """Attributes
    ---------------
    w_ : 1d-array
        weights after fitting
    b_ : Scalar
        Bias unit after fitting
    """

    def __init__(self, eta=0.01, n_itr=50, random_state=1) -> None:
        """_summary_

        Args:
            eta (float, optional): Learning rate. Defaults to 0.01.
            n_itr (int, optional): number of iterations. Defaults to 50.
            random_state (int, optional): Initialization variable. Defaults to 1.
        """
        self.eta = eta
        self.n_itr = n_itr
        self.random_state = random_state

    def fit(self, X, y):

        random_generator = np.random.RandomState(self.random_state)

        """Initialize parameters
        """
        self.w_ = random_generator.normal(loc=0.0, scale=0.01,
                                          size= X.shape[1])
        self.b_ = np.float_(0)
        self.errors_ = []

        """Training loop
        """
        for _ in range(self.n_itr):
            errors = 0
            for xi,target in zip(X,y):
                update = self.eta * (target - self.predict(xi))

                """Update network parameters
                """
                self.w_ += update * xi
                self.b_ += update

                """Accumulate error for each trainig loop
                """
                errors += int(update != 0.0)
            

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """Forward propagates through the network

        Args:
            X (_type_): input array

        Returns:
            _type_: network outout array
        """
        return np.dot(X, self.w_) + self.b_
    
    def get_errors(self):
        return self.errors_
    
    def predict(self, X):
        """Predicts class of the given input based on threshold

        Args:
            X (_type_): input data array

        Returns:
            _type_: predicted class of the input data array
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    

