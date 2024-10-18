import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
import random


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
            
        self.save_samples(X, y)
            
        
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
    
    def save_samples(self, X, y):
        self.Xs_dict = {i: [] for i in range(10)}
        for i in range(len(y)):
            self.Xs_dict[y[i]].append(X[i])
        for key in self.Xs_dict:
            self.Xs_dict[key] = np.array(self.Xs_dict[key])
            
    def decide_class(self, Y):
        if isinstance(Y, int):
            return Y
        elif Y == 'default':
            Y_values = list(range(10))
            random_number = np.random.choice(Y_values, size=1, p=self.P_y)
            gen_Y = random_number.item()
            return gen_Y
        else:
            Y_values = list(range(10))
            random_number = np.random.choice(Y_values, size=1, p=Y)
            gen_Y = random_number.item()
            return gen_Y
            
    
    def generate_img(self, Y):
        Y = self.decide_class(Y)
        random_matrix = np.random.rand(28, 28)
        x_prob = self.P_x_y[2 * Y + 1].reshape(28, 28)
        img = x_prob * random_matrix 
        out_img = self.improve_img(img, 0.12)
        return out_img, Y
    
    def generate_img2(self, Y):
        Y = self.decide_class(Y)
        random_matrix1 = np.random.rand(28, 28)
        random_matrix2 = np.random.rand(28, 28)
        sample_i = random.randint(0, self.Xs_dict[Y].shape[0]-1)
        sample_x = (self.Xs_dict[Y][sample_i].reshape(28,28) + 0.01) * random_matrix1
        x_prob = self.P_x_y[2 * Y + 1].reshape(28, 28)
        img = x_prob * random_matrix2 * sample_x
        out_img = self.improve_img(img, 0.03)
        return out_img, Y
    
    def improve_img(self, img, threshold):
        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        img_binary = np.where(blur_img > threshold, 1, 0)
        skeleton = morphology.skeletonize(img_binary).astype(np.float32)
        blur_img_2 = cv2.blur(skeleton, (2, 2))
        img_bright = blur_img_2 * 2
        img_bright = np.where(img_bright > 1, 1, img_bright)
        return img_bright


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
    axes[i].set_title(f"Predict:{y_pred[i]}  Probability:{y_prob[i]:.2f}", fontsize = 12)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

Y1 = 3
Y2 = [0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
Y3 = 'default'

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()
for i in range(10):
    gen_img, gen_Y = nb_classifier.generate_img2(Y3)
    axes[i].imshow(gen_img, cmap='gray')
    axes[i].set_title(f"Label: {gen_Y}", fontsize = 12)
    axes[i].axis('off')
plt.tight_layout()
plt.show()




















