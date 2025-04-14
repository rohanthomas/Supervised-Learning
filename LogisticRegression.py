import numpy as np

class LogisticRegression():
    """
    A class performing Binary Logistic Regression. The decision boundaries are assumed to be linear.

    Example:
        >>> import numpy as np
        >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        >>> y_train = np.array([0, 0, 0, 1, 1, 1])
        >>> epochs, alpha = 1000, 0.01
        >>> model = LogisticRegression(X_train, y_train, epochs, alpha)
        >>> w_0, b_0 = np.zeros(X_train.shape[1]), 0
        >>> w, b = model.train(w_0, b_0)
        >>> x_predict = np.array([[2, 3], [4, 5], [6, 7]])
        >>> predictions = model.predict(x_predict, w, b)
        >>> print(predictions)
    """
    
    def __init__(self, X_train, y_train, epochs, alpha):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.alpha = alpha
    
    def _sigmoid(self, z):
        g = 1 / (1 + np.exp(-z))
        return g
    
    def _gradient(self, w, b):
        m = self.X_train.shape[0]

        z = np.dot(self.X_train, w) + b
        f = self._sigmoid(z)

        error = (f - self.y_train)
        dj_dw = 1/m * np.dot(self.X_train.T, error)
        dj_db = 1/m * np.sum(error)

        return dj_dw, dj_db  
    
    def _cost_function(self, w, b):
        m, n = self.X_train.shape

        z = np.dot(self.X_train, w) + b
        f = self._sigmoid(z)
    
        loss = self.y_train * np.log(f) + (1 - self.y_train) * np.log(1 - f)
        cost = np.sum(loss)
        
        total_cost = -1/m * cost

        return total_cost
    
    def train(self, w_0 = None, b_0 = 0):
        if w_0 is None:
            w_0 = np.zeros(self.X_train.shape[1])
            
        w, b = w_0, b_0

        for i in range(self.epochs):
            dj_dw, dj_db = self._gradient(w, b)
            w = w - self.alpha * dj_dw
            b = b - self.alpha * dj_db

            cost = self._cost_function(w, b)
            if i % 10 == 0:
                print(f"Epoch: {i} Cost: {cost}")
        return w, b
    
    def predict(self, x_predict, fitted_w, fitted_b):
        m = x_predict.shape[0]
        p = np.zeros(m)
        z = np.dot(x_predict, fitted_w) + fitted_b
        f = self._sigmoid(z)

        for i in range(m):
            if f[i] > 0.5:
                p[i] = 1
            else:
                p[i] = 0
        return p