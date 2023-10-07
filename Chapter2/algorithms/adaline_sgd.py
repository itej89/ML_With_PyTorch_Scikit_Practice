import numpy as np

class AdalineSGD:
    """AdalineSGD Classifier
    """


    """Attributes
    ---------------
    w_ : 1d-array
        weights after fitting
    b_ : Scalar
        Bias unit after fitting
    """

    def __init__(self, eta=0.01, n_itr=50, random_state=1, shuffle=True) -> None:
        """_summary_

        Args:
            eta (float, optional): Learning rate. Defaults to 0.01.
            n_itr (int, optional): number of iterations. Defaults to 50.
            random_state (int, optional): Initialization variable. Defaults to 1.
            shuffle (float, optional): Shuffle data. Defaults to True.
        """
        self.eta = eta
        self.n_itr = n_itr
        self.random_state = random_state
        self.shuffle = True
        self.w_initialized = False

    def fit(self, X, y):
        """Fits entire training data

        Args:
            X (np array): input train X
            y (np array): input target

        Returns:
            _type_: _description_
        """
        self.initialize_weights(X.shape[1])
        self.errors_ = []
        for i in range(self.n_itr):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self.update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.errors_.append(avg_loss)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self.initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self.update_weights(xi, target)
        else:
            self.update_weights(X, y)
        return self


    def _shuffle(self, X, y):
        """Shuffle data

        Args:
            X (np array): train X
            y (np array): train y

        Returns:
            tuple: Shuffled X and y
        """
        r = self.random_generator.permutation(len(y))
        return X[r], y[r]
    
    def initialize_weights(self, m):
        """intializes weights

        Args:
            m (_type_): shape of weights parameter
        """
        self.random_generator = np.random.RandomState(self.random_state)
        self.w_ = self.random_generator.normal(loc=0.0, scale=0.01,
                                          size= m)
        self.b_ = np.float_(0)
        self.w_initialized = True

    def update_weights(self, xi ,target):
        """method update weight for a given training example set

        Args:
            xi (np array): one training example
            target (np array): target vale

        Returns:
            _type_: _description_
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)

        """Update the weights and biases

        Returns:
            loss: squared loss
        """
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    
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
        return self.errors_