import numpy as np

class MultipleLinearRegression():
    """
    A class performing multiple linear regression.

    Example:
        >>> import numpy as np
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y_train = np.array([7, 8, 9])
        >>> epochs, alpha = 1000, 0.01
        >>> model = MultipleLinearRegression(X_train, y_train, epochs, alpha)
        >>> w_0, b_0 = np.zeros(X_train.shape[1]), 0
        >>> w, b = model.train(w_0, b_0)
        >>> x_predict = np.array([7, 8])
        >>> prediction = model.predict(x_predict, w, b)
        >>> print(prediction)
        
    """
    def __init__(self, X_train, y_train, epochs, alpha):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.alpha = alpha

    def _gradient(self, w, b):
        """
        Computes gradient of the cost function.

        Args:
            w (numpy.ndarray): Weights
            b (float): Bias.

        Returns:
            tuple: 
            - djdw (numpy.ndarray): A vector containing the partial derivatives of the cost function with respect to each weight, shape (n_features,).
            - djdb (float): The partial derivative of the cost function with respect to bias.
        """
        m = self.X_train.shape[0]

        error = (np.dot(w, self.X_train.T) + b - self.y_train)  # Each iteration of the sum is collected into a vector
        djdw = 1/m * np.dot(self.X_train.T, error)  # Does the usual matrix-vector product
        djdb = 1/m * np.sum(error)
        
        return djdw, djdb
    
    def _cost_function(self, w, b):
        """
        Computes mean square cost function.

        Args:
            w (numpy.ndarray): Weights
            b (float): Bias.

        Returns:
            cost (float): Cost.
        """
        m = self.X_train.shape[0]
        cost = 1/(2*m) * np.sum(np.square((np.dot(w, self.X_train.T) + b - self.y_train)))
        return cost
    
    def train(self, w_0 = 0, b_0 = 0):
        """
        Trains the model using gradient descent.

        Args:
            w_0 (numpy.ndarray or float): Initial weights.
            b_0 (float): Initial bias.

        Returns:
            tuple:
            - w (numpy.ndarray): Fitted weights.
            - b (float): Fitted bias.
        """
        w, b = w_0, b_0

        for i in range(self.epochs):
            djdw, djdb = self._gradient(w, b)
            w = w - self.alpha * djdw
            b = b - self.alpha * djdb

            cost = self._cost_function(w, b)
            if i % 10 == 0:
                print(f"Epoch: {i} Cost: {cost}")
        return w, b
    
    def predict(self, x_predict, fitted_w, fitted_b):
        """
        Predicts the expected value.

        Args:
            x_predict (numpy.ndarray): Input values to predict 
            fitted_w (numpy.ndarray): Fitted weights from train() method.
            fitted_b (float): Fitted bais from train() method

        Returns:
            predicted (float): Predicted output for the given input values.
        """
        predicted = np.dot(fitted_w, x_predict) + fitted_b
        return predicted
