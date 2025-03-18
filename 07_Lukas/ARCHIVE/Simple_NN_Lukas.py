import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
from collections import Counter
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class SimpleNN:
    """
    A simple fully connected neural network with two hidden layers.
    Implements forward propagation, backward propagation, and gradient descent.
    """
    def __init__(self, input_size, hidden_size1, hidden_size2):
        np.random.seed(0)  # For reproducibility
        self.w1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        
        self.w2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        
        self.w3 = np.random.randn(hidden_size2, 1) * 0.01
        self.b3 = np.zeros((1, 1))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.maximum(0, self.z2)  # ReLU
        
        self.z3 = np.dot(self.a2, self.w3) + self.b3  # Linear output for regression
        return self.z3

    def backward(self, X, y, learning_rate=0.01):
        m = y.shape[0]

        d_loss = self.z3 - y  # (batch_size, 1)
        d_z3 = d_loss.reshape(-1, 1) / m  # Ensure correct shape

        d_w3 = np.dot(self.a2.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        d_a2 = np.dot(d_z3, self.w3.T)
        d_z2 = d_a2 * (self.z2 > 0)  # ReLU derivative
        d_w2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.w2.T)
        d_z1 = d_a1 * (self.z1 > 0)  # ReLU derivative
        d_w1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights
        self.w1 -= learning_rate * d_w1
        self.b1 -= learning_rate * d_b1
        self.w2 -= learning_rate * d_w2
        self.b2 -= learning_rate * d_b2
        self.w3 -= learning_rate * d_w3
        self.b3 -= learning_rate * d_b3

    def train(self, X, y, epochs=500, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y) ** 2)  # MSE loss
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")