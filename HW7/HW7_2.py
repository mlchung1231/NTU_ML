import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, eta=0.1, iterations=2000):
    N = X.shape[0]
    w = np.zeros(X.shape[1])

    for i in range(iterations):
        gradient = (X * y[:, np.newaxis]) * (sigmoid(-y*(X @ w))[:, np.newaxis])
        gradient = np.sum(gradient, axis=0) / N
        w += eta * gradient                 
        
        if i % 100 == 0:
            loss = abs(y - (2 * sigmoid(X @ w) -1))
            loss = np.sum(loss)
            print(f"Iteration {i}, Loss: {loss}")

    return w


def predict(X, weights):
    X = np.c_[X, np.ones(X.shape[0])]  
    probabilities = sigmoid(X @ weights)
    return probabilities


data = pd.read_csv('Data_for_LR.csv')

X = data.iloc[:,0].values
y = data.iloc[:,1].values

X = np.c_[X, np.ones(X.shape[0])]  
initial_w = np.zeros(X.shape[1])


GD_weights = gradient_descent(X, y)
print("GD_weights: ", GD_weights)

X_test = np.array([1.5])
predictions = predict(X_test, GD_weights)
print('\nstudy hours:',X_test)
print('pass probabilities:',predictions)


X_plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
predictions = predict(X_plot, GD_weights)
y_sample_plot = np.where(y == -1, 0 , 1)

plt.figure(figsize=(10, 6))
plt.scatter(data.iloc[:,0], y_sample_plot, color='blue', label='Data Points')
plt.plot(X_plot, predictions, color='red', label='Pass Probability')
plt.xlabel('Study Hours')
plt.ylabel('Pass Probability')
plt.title('Logistic Regression Curve')
plt.legend()
plt.grid()
plt.show()