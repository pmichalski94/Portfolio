import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
np.random.seed(42)

import warnings
warnings.filterwarnings('ignore') #suppress warnings

class Logistic_Regression_SGD_momentum():
    def __init__(self):
        self.w = np.ndarray
        self.B = np.ndarray
    
    #Sigmoid function
    def __S(self, t):

        S = 1/(1 + np.exp(-t))
        return S
    
    #Lerning schedule
    def __learning_schedule(self,t0, t1, t):
        return t0/(t+t1)
    
    def fit(self,X: np.ndarray, y: np.ndarray, n_iter, learning_schedule=(5,50), momentum_beta=0.9) -> None:
        n_features = X.shape[1]
        # Initialization of initial weights and biases are 0 
        B = 0 
        w = np.zeros(n_features)
        
        # Initialization of initial moment vectors are 0
        m_w = 0
        m_B = 0
        beta = momentum_beta
        
        #Learning schedule hyperparameters
        t0, t1 = learning_schedule
      
        for n in range(n_iter): 
            for i in range(X.shape[0]): 
                random_index = np.random.randint(X.shape[0]) 
                Xi = X[random_index:random_index+1] 
                yi = y[random_index:random_index+1]
                

                Z = np.dot(w, np.transpose(Xi)) + B
                class_pred = self.__S(Z)
                dZ = class_pred - yi

                dw = np.dot(np.transpose(Xi), dZ)
                dB = np.sum(dZ)
                
                #Learning rate with simulated annealing
                lr = self.__learning_schedule(t0, t1, n*X.shape[0]+i) 

                
                m_w = beta*m_w - lr*dw
                w += m_w
                
                m_B = beta*m_B - lr*dB
                B += m_B
    
        self.w = w
        self.B = B
        
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.w.dot(X.T) + self.B
        pred = []
        for i in range(len(self.__S(Z))):
            if self.__S(Z)[i] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_true = 0

        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                true_true += 1 
        return true_true / len(y_true)



def train_model(number: str):
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]

    x_train, x_test, y_all_digits, y_test_all_digits = X[:60000], X[60000:], y[:60000], y[60000:]

    def change_labels(dataset):
        dataset = np.where(dataset == number, 1, 0)
        return dataset

    y_train = change_labels(y_all_digits)
    y_test = change_labels(y_test_all_digits)

    log_reg = Logistic_Regression_SGD_momentum()
    log_reg.fit(x_train, y_train.ravel(), n_iter = 2, learning_schedule=(5, 500))
    y_pred = log_reg.predict(x_test)

    return x_train, x_test, y_train, y_test, y_pred, log_reg.evaluate(y_test, y_pred)

    