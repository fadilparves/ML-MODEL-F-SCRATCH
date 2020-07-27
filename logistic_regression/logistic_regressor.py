import numpy as np

class LogisticRegressor:

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def predict_probs(self, X):
        X = self._add_intercept(X) 
        return self._sigmoid(np.dot(X, self.W))

    def predict(self, X):
        return self.predict_probs(X).round()

    def fit(self, X, y, n_iter=100000, lr=0.01):
        X = self._add_intercept(X) 
        self.W = np.zeros(X.shape[1])

        for i in range(n_iter):
            z = np.dot(X, self.W)
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size 
            self.W -= lr * gradient
            
        return self