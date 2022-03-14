import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from visualizer import visualize_svm

class SimpleSVM:
    def __init__(self, lr=0.001, lambda_p=0.01, iter_num=1000):
        # define main parameters
        self.lr = lr
        self.lambda_p = lambda_p
        self.iter_num = iter_num
        
        # define weights and bias
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        # get number of samples and number of features
        n_samples, n_features = X.shape
        # replace values lower than or equals zero to -1, otherwise 1
        y_sign = np.where(y <= 0, -1, 1)
        # initialize model's parameters
        self.w = np.zeros(n_features)
        self.b = 0
        
        # start leaning the model
        for _ in range(self.iter_num):
            for idx, sample in enumerate(X):
                # check if y * (W.X - b) >= 1 
                check = y_sign[idx] * (np.dot(self.w, sample) - self.b) >= 1
                # update model's parameter
                if check:
                    self.w -= (2 * self.lambda_p * self.w) * self.lr
                else:
                    self.w -= ((2 * self.lambda_p * self.w) - np.dot(y_sign[idx], sample)) * self.lr
                    self.b -= y[idx] * self.lr
                
    def predict(self, X):
        # simply substitude in the equation (W.X - b):
        #  1) if it's positive == > assign it to class 1
        #  1) if it's negative == > assign it to class 0
        
        out = np.dot(self.w, X) - self.b
        # we can use np sign function to get the sign of number easily
        return np.sign(out)

    
# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # create dataset
    X, y = make_blobs(
        n_samples=300, 
        n_features=2, 
        centers=2, 
        cluster_std=1.5
    )
    y = np.where(y == 0, -1, 1)

    # train a model
    clf = SimpleSVM()
    clf.fit(X, y)
    
    
    # visualize results
    visualize_svm(X, y, clf)