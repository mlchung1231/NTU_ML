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
    if train_Y[i][0] == '(1.0)':
        train_Y[i][0] = -1.0

train_Y = train_Y.astype(float)

model = myPLA()
model.fit(train_X, train_Y)

#prediction = model.predict([4.5, 3])
#print("Predictions:", prediction[0])

label_1_dist = model.margin(train_X[0:50])
margins_1 = min(label_1_dist)

label_neg1_dist = model.margin(train_X[50:100])
margins_neg1 = min(label_neg1_dist)


margin_1_index = np.argmin(label_1_dist)
margin_neg1_index = np.argmin(label_neg1_dist) + 50

unit_w = model.weights / np.linalg.norm(model.weights)


P_1 = train_X[margin_1_index] - margins_1 * unit_w.T
P_neg1 = train_X[margin_neg1_index] + margins_neg1 * unit_w.T
P_1 = P_1.reshape(2,)
P_neg1 = P_neg1.reshape(2,)

plt.figure(figsize=(8, 6))
plt.title('HW3_3 PLA classifier',fontsize=20)
plt.axis('equal')
plt.xlabel('X1',fontsize=16)
plt.ylabel('X2',fontsize=16)
plt.scatter(train_X[0:50][:, 0], train_X[0:50][:, 1], color='b', label='Class 1', s=15)
plt.scatter(train_X[50:100][:, 0], train_X[50:100][:, 1], color='r', label='Class -1', s=15)
plt.xlim(1, 8)
plt.ylim(0.5, 5.5)
x_min, x_max = plt.xlim()
y_min = -(model.weights[0] / model.weights[1]) * x_min - (model.bias / model.weights[1])
y_max = -(model.weights[0] / model.weights[1]) * x_max - (model.bias / model.weights[1])
plt.plot([x_min, x_max], [y_min, y_max], color='g', label='PLA classifier')
plt.plot([train_X[margin_1_index][0], P_1[0]], [train_X[margin_1_index][1], P_1[1]], color='c', linestyle='--', label='margin of class 1')
plt.plot([train_X[margin_neg1_index][0], P_neg1[0]], [train_X[margin_neg1_index][1], P_neg1[1]], color='m', linestyle='--', label='margin of class -1')
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.2)
plt.legend()
plt.show()

print(f"margin of class 1： {margins_1[0]:.4f}")
print(f"margin of class -1： {margins_neg1[0]:.4f}")