import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class myPLA:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, train_X, train_Y):
        self.weights = np.zeros((train_X.shape[1], 1))
        self.bias = 0

        i = 0
        while(i < train_X.shape[0]):
            x = train_X[i,0:2].reshape(2,1)
            if train_Y[i] * (self.weights.T @ x) <= 0:
                self.weights += train_Y[i] * x
                self.bias += train_Y[i]
                i = 0
            i += 1

    def predict(self, x):
        predicted_y = np.sign((self.weights.T @ x) + self.bias)
        
        return predicted_y
    
    def margin(self, train_X):
        distances = np.abs(train_X @ self.weights + self.bias) / np.linalg.norm(self.weights)
        return distances



dataset = pd.read_csv('./HW3_PLA_1.csv',header=None)
train_X = dataset.iloc[:, 0:2].values
train_Y = dataset.iloc[:, 2:3].values

for i in range(train_Y.shape[0]):
    #a = train_Y[i][0]
    #print(a)
    if train_Y[i][0] == '(1.0)':
        train_Y[i][0] = -1.0

train_Y = train_Y.astype(float)

model = myPLA()
model.fit(train_X, train_Y)

#prediction = model.predict([4.5, 3])
#print("Predictions:", prediction[0])


margins_1 = min(model.margin(train_X[0:50]))
print(f"margin of class 1： {margins_1[0]:.4f}")
margins_neg1 = min(model.margin(train_X[50:100]))
print(f"margin of class -1： {margins_neg1[0]:.4f}")



plt.figure(figsize=(8, 6))
plt.title('HW3_3 PLA classifier',fontsize=20)
plt.xlabel('X1',fontsize=16)
plt.ylabel('X2',fontsize=16)
plt.scatter(train_X[0:50][:, 0], train_X[0:50][:, 1], color='blue', label='Class 1', s=15)
plt.scatter(train_X[50:100][:, 0], train_X[50:100][:, 1], color='red', label='Class -1', s=15)
plt.xlim(4, 8)
plt.ylim(0, 6)
x_min, x_max = plt.xlim()
y_min = -(model.weights[0] / model.weights[1]) * x_min - (model.bias / model.weights[1])
y_max = -(model.weights[0] / model.weights[1]) * x_max - (model.bias / model.weights[1])
plt.plot([x_min, x_max], [y_min, y_max], color='green', label='PLA classifier')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
plt.legend()
plt.show()