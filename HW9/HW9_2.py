import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - np.tanh(x)**2   
    
def add_layers(input_size, output_size):
    w = np.random.randn(input_size, output_size)
    b = np.random.randn(1, output_size)
    return w, b

def forward(X, w2, b2, w3, b3):
    z2 = X @ w2 + b2
    a2 = tanh(z2)
    z3 = a2 @ w3 + b3
    a3 = tanh(z3)
    return a3, a2
    

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

eta = 0.3
hidden_size = [2]
epoch = 800

w2, b2 = add_layers(X.shape[1], hidden_size[0])
z2 = X @ w2 + b2
a2 = tanh(z2)

w3, b3 = add_layers(hidden_size[0], Y.shape[1])
z3 = a2 @ w3 + b3
a3 = tanh(z3)

W_history = np.zeros((epoch, 9))
W_history[0,:] = [w2[0,0], w2[1,0], w2[0,1], w2[1,1],
                  w3[0,0], w3[1,0],
                  b2[0,0], b2[0,1], b3[0,0]]

Loss_history = []
Loss_history.append(np.mean(np.square(Y - a3)))
    
for i in range(epoch):
    
    z2 = X @ w2 + b2
    a2 = tanh(z2)
    z3 = a2 @ w3 + b3
    a3 = tanh(z3)
    
    error = Y - a3
    de3 = tanh_derivative(z3) * error
    d_w3 = a2.T @ de3
    d_b3 = np.sum(de3, axis=0, keepdims=True)
    
    de2 = tanh_derivative(z2) * (de3 @ w3.T)
    d_w2 = X.T @ de2
    d_b2 = np.sum(de2, axis=0, keepdims=True)
    
    w3 += eta * d_w3
    b3 += eta * d_b3
    w2 += eta * d_w2
    b2 += eta * d_b2
    
    W_history[i,:] = [w2[0,0], w2[1,0], w2[0,1], w2[1,1],
                      w3[0,0], w3[1,0],
                      b2[0,0], b2[0,1], b3[0,0]]
    
    Loss_history.append(np.mean(np.square(error)))
    
    if i%100 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {i},  Loss: {loss:.4f}")

print("\npredict:")
print(a3)
        
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
param_names = [
    "W1", "W2", "W3", "W4",
    "W5", "W6",
    "W7", "W8", "W9"]

for i, ax in enumerate(axs.flat):
    ax.plot(range(epoch), W_history[:, i])
    ax.set_title(param_names[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
plt.plot(Loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss History")
plt.grid(True)
plt.show()


xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

a3_grid, _ = forward(grid_points, w2, b2, w3, b3)
zz = a3_grid.reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap='RdYlBu', alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='RdYlBu', edgecolors='k', marker='o', s=100)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Decision Boundary')
plt.axis('equal')
plt.show()
        
