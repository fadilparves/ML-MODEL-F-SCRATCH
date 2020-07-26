import numpy as np
class LinearRegression:
    
    def predict(self, X):
        return np.dot(X, self._W)
    
    def _gradientDescentStep(self, X, targets, lr):
        
        y_hat = self.predict(X)
        
        error = y_hat - targets
        gradient = np.dot(X.T, error) / len(X)
        
        self._W -= lr * gradient
        
    def fit(self, X, y, n_iter=100000, lr=0.01):
        
        self._W = np.zeros(X.shape[1])
        
        self._cost_history = []
        self._W_history = [self._W]
        
        for i in range(n_iter):
            
            prediction = self.predict(X)
            cost = loss(prediction, y)
            
            self._cost_history.append(cost)
            
            self._gradientDescentStep(X, y, lr)
            
            self._W_history.append(self._W.copy())
            
        return self