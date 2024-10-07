import numpy as np
from tensorflow.keras.datasets import mnist

class myNaiveBayesClassifier:

    def fit(self, X, y):
        self.classes, self.y_counts = np.unique(y, return_counts=True)
        self.P_y = self.y_counts / y.shape[0]
        self.c_x_y = np.zeros((len(self.classes), X.shape[1]));
        self.P_x_y = np.zeros((2 * len(self.classes), X.shape[1]));
        for n in range(len(y)):
            self.c_x_y[y[n]] += X[n]
        for c in range(len(self.classes)):
            self.P_x_y[2*c + 1] = self.c_x_y[c] / self.y_counts[c]
            self.P_x_y[2*c] = 1 - self.P_x_y[2*c + 1]
            
        
    def predict(self, X):
        predictions = []
        for n in range(X.shape[0]):
            P_y_x = []
            for c in range(len(self.classes)):
                temp_1 = X[n] * self.P_x_y[2 * c + 1]
                temp_0 = (1 - X[n]) * self.P_x_y[2 * c]
                temp = temp_0 + temp_1
                P_xs_y = np.prod(temp)
                P_y_x.append(P_xs_y * self.P_y[c])
            max_class = np.argmax(P_y_x)
            predictions.append(max_class)
        return np.array(predictions)



(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flat = x_test.reshape(-1, 28 * 28) / 255.0

threshold = 0.2
x_train_binary = (x_train_flat > threshold).astype(np.float32)
x_test_binary = (x_test_flat > threshold).astype(np.float32)


nb_classifier = myNaiveBayesClassifier()
nb_classifier.fit(x_train_binary, y_train)


y_pred = nb_classifier.predict(x_test_binary)


accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")



