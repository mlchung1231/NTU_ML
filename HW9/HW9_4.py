import numpy as np
import matplotlib.pyplot as plt
import string
from collections import Counter
from sklearn.model_selection import train_test_split
import seaborn as sns


word_count = Counter()
all_words = []
all_labels = []


#------------------------------------------------
#            Data Preprocessing
#------------------------------------------------

with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        
        if '\t' in line:
            label, sentence = line.split('\t', 1)
            
            sentence = sentence.lower()
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))
            
            words = sentence.split()
            
            word_count.update(words)
            all_words.extend(words)
            
            all_labels.append(label)

unique_words = sorted(set(all_words))
word_to_index = {word: idx for idx, word in enumerate(unique_words)}

def sentence_to_vector(sentence, word_to_index):
    vector = [0] * len(word_to_index)
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = sentence.split()
    for word in words:
        if word in word_to_index:
            index = word_to_index[word]
            vector[index] = 1
    
    return vector

vectors = []
with open('SMSSpamCollection.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        
        if '\t' in line:
            label, sentence = line.split('\t', 1)
            vector = sentence_to_vector(sentence, word_to_index)
            vectors.append(vector)

sentence_vector = np.array(vectors)

"""
sample = np.where(sentence_vector[1,:] == 1)

for w in sample[0]:
    print(unique_words[w])
"""

X = sentence_vector
Y = np.array(all_labels)
Y = np.where(Y == 'spam', 1, 0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



#------------------------------------------------
#            Logistic Regression
#------------------------------------------------

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epoches=1000, batch=32, decay=1):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.w = None
        self.b = None
        self.batch_size = batch
        self.decay = 1

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        m = X.shape[0]
        predictions = self.sigmoid(np.dot(X, self.w) + self.b)
        loss = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.epoches):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for j in range(0, m, self.batch_size):
                X_batch = X_shuffled[j:j+self.batch_size]
                y_batch = y_shuffled[j:j+self.batch_size]

                predictions = self.sigmoid(np.dot(X_batch, self.w) + self.b)

                dw = (1/self.batch_size) * np.dot(X_batch.T, (predictions - y_batch))  # w 的梯度
                db = (1/self.batch_size) * np.sum(predictions - y_batch)  # b 的梯度

                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                
                self.learning_rate *= self.decay

            if i % 10 == 0:
                loss = self.compute_loss(X, y)
                print(f"Epoch {i}: Loss = {loss}")

    def predict(self, X):
        predictions = self.sigmoid(np.dot(X, self.w) + self.b)
        return (predictions >= 0.5).astype(int) 

LR_model = LogisticRegression(learning_rate=2, epoches=150, batch=32, decay=0.9)
LR_model.fit(X_train, Y_train)
print('\n')
    
LR_predictions = LR_model.predict(X_test)
LR_accuracy = np.mean(Y_test == LR_predictions)
print(f"Logistic Regression accuracy: {LR_accuracy}")



#------------------------------------------------
#            Naive Bayes
#------------------------------------------------

class NaiveBayes:
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
            

NB_model = NaiveBayes()
NB_model.fit(X_train, Y_train)

NB_predictions, NB_prob = NB_model.predict(X_test)
NB_accuracy = np.mean(NB_predictions == Y_test)
print(f"Naive Bayes accuracy: {NB_accuracy}")


#------------------------------------------------
#            Confusion Matrix
#------------------------------------------------

def plot_confusion_matrix(Y_true, Y_pred):
    add_YY = Y_pred + Y_true
    sub_YY = Y_pred - Y_true
    TP = np.count_nonzero(add_YY == 2)
    TN = np.count_nonzero(add_YY == 0)
    FP = np.count_nonzero(sub_YY == 1)
    FN = np.count_nonzero(sub_YY == -1)
    cm = np.array([[TN, FP], [FN, TP]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
plot_confusion_matrix(Y_test, LR_predictions)
plot_confusion_matrix(Y_test, NB_predictions)



#------------------------------------------------
#            Test
#------------------------------------------------
test1 = 'You are a winner U have been specially selected 2 receive £1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810810'
test2 = 'I just reached home. I go bathe first. But my sis using net tell u when she finishes k...'
test = [test1, test2] 

test_vectors = []
for line in test:
    line = line.strip()
    
    test_vector = sentence_to_vector(line, word_to_index)
    test_vectors.append(test_vector)

test_X = np.array(test_vectors)

LR_pred_test = LR_model.predict(test_X)
NB_pred_test, NB_prob = NB_model.predict(test_X)

LR_pred_test = np.where(LR_pred_test == 1, 'spam', 'ham')
NB_pred_test = np.where(NB_pred_test == 1, 'spam', 'ham')

print('\ntest data in (c):')
print(f"Logistic Regression predictions: {LR_pred_test}")
print(f"Naive Bayes predictions: {NB_pred_test}")
