import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2


class myNaiveBayesClassifier:

    def fit(self, X, y):
        self.classes, self.y_counts = np.unique(y, return_counts=True)
        self.P_y = self.y_counts / y.shape[0]
        self.c_x_y = np.zeros((len(self.classes), X.shape[1]));
        self.P_x_y = np.zeros((2 * len(self.classes), X.shape[1]));
        for n in range(len(y)):
            self.c_x_y[y[n]] += X[n]
        self.c_x_y += 1 
        for c in range(len(self.classes)):
            self.P_x_y[2*c + 1] = self.c_x_y[c] / (self.y_counts[c] + 1)
            self.P_x_y[2*c] = 1 - self.P_x_y[2*c + 1]
            
        
    def predict(self, X):
        predictions = []
        probability = []
        for n in range(X.shape[0]):
            P_y_x = []
            for c in range(len(self.classes)):
                temp_1 = X[n] * self.P_x_y[2 * c + 1]
                temp_0 = (1 - X[n]) * self.P_x_y[2 * c] 
                temp = np.log(temp_0 + temp_1)
                P_xs_y_log = np.sum(temp)
                P_xs_y = np.exp(P_xs_y_log)
                P_y_x.append(P_xs_y * self.P_y[c])
            max_class = np.argmax(P_y_x)
            P_x = np.sum(P_y_x)
            predictions.append(max_class)
            probability.append(P_y_x[max_class] / P_x)
        return np.array(predictions), np.array(probability)
    
    def plot_P_x_y(self):
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        axes = axes.ravel()
        for i in range(10):
            img_P_x_y = self.P_x_y[2 *i + 1].reshape(28,28)
            axes[i].imshow(img_P_x_y, cmap='gray')
            axes[i].set_title(f"Label: {i}", fontsize = 12)
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flat = x_test.reshape(-1, 28 * 28) / 255.0

threshold = 0.2
x_train_binary = (x_train_flat > threshold).astype(np.float32)
x_test_binary = (x_test_flat > threshold).astype(np.float32)

nb_classifier = myNaiveBayesClassifier()
nb_classifier.fit(x_train_binary, y_train)
nb_classifier.plot_P_x_y()

y_pred, y_prob = nb_classifier.predict(x_test_binary)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
for i in range(10):
    axes[i].imshow(x_test[i], cmap='gray')
    axes[i].set_title(f"Real:{y_test[i]}, Predict:{y_pred[i]}", fontsize = 12)
    axes[i].axis('off')
plt.tight_layout()
plt.show()


images = [
    cv2.imread('./0.jpg', cv2.IMREAD_GRAYSCALE),
    cv2.imread('./2.jpg', cv2.IMREAD_GRAYSCALE),
    cv2.imread('./4.jpg', cv2.IMREAD_GRAYSCALE),
]

images_label = [0, 2, 4]


fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes = axes.ravel()

fix_image = np.zeros((3,28,28))
for i in range(len(images)):
    height, width = images[i].shape[:2]
    if height > width:
        delta_w = height - width
        left = delta_w // 2
        right = delta_w - left
        top = 0
        bottom = 0
    else:
        delta_h = width - height
        top = delta_h // 2
        bottom = delta_h - top
        left = 0
        right = 0

    square_image = cv2.copyMakeBorder(images[i], top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    fix_image[i] = cv2.resize(square_image, (28, 28))
    
    axes[i].imshow(fix_image[i], cmap='gray')
    axes[i].set_title(f"{images_label[i]}.jpg after resize", fontsize = 12)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

fix_image_flat = fix_image.reshape(-1, 28 * 28) / 255.0
fix_image_binary = (fix_image_flat > threshold).astype(np.float32)
img_pred, img_prob = nb_classifier.predict(fix_image_binary)

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes = axes.ravel()
for i in range(3):
    axes[i].imshow(fix_image[i], cmap='gray')
    axes[i].set_title(f"Predict:{img_pred[i]}  Probability:{img_prob[i]:.4f}", fontsize = 12)
    axes[i].axis('off')
plt.tight_layout()
plt.show()









