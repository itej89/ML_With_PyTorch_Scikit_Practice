import numpy as np

class AdalineGD:
    """AdalineGD Classifier
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
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = y-output

            self.w_ += self.eta * (2/X.shape[0])*X.T.dot(errors)
            self.b_ += self.eta * 2.0 * errors.mean()

            loss = (errors**2).mean()

            self.errors_.append(loss)

        return self
    
    def activation(self, X):
        """In Adaline activation is an Identity funciton

        Args:
            X (np array): input network output

        Returns:
            np array: same as input
        """
        return X

    def net_input(self, X):
        """Forward propagates through the network

        Args:
            X (np array): input array

        Returns:
            np array: network outout array
        """
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Predicts class of the given input based on threshold

        Args:
            X (np array): input data array

        Returns:
            np array: predicted class of the input data array
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
        

    def get_errors(self):
        return np.log10(self.errors_)