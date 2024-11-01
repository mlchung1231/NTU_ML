import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, data=None, score=None):
        self.feature = feature  
        self.threshold = threshold  
        self.left = left        
        self.right = right            
        self.data = data

class DecisionTree:
    def __init__(self):
        self.root = None
        self.score_list = []

    def fit(self, df):
        self.root = self._build_tree(data)

    def _build_tree(self, data):
        root_node = DecisionTreeNode(feature=0, threshold=1.5, data=data)

        # 左子樹 (Outlook <= 1.5)
        Root_L_indices = data[:, 0] <= 1.5
        Root_L_node = DecisionTreeNode(feature=1, threshold=1.5, data=data[Root_L_indices])
        root_node.left = Root_L_node

        # 右子樹 (Outlook > 1.5)
        Root_R_node = DecisionTreeNode(feature=2, threshold=0.5, data=data[~Root_L_indices])
        root_node.right = Root_R_node
        #print(data[~Root_L_indices])

        # 左左子樹 (Humidity <= 65)
        Root_LL_indices = Root_L_node.data[:, 1] <= 65
        Root_L_node.left = DecisionTreeNode(data=Root_L_node.data[Root_LL_indices])
        #print(Root_L_node.data[Root_LL_indices])

        # 左右子樹 (Humidity > 65)
        Root_L_node.right = DecisionTreeNode(data=Root_L_node.data[~Root_LL_indices])
        #print(Root_L_node.data[~Root_LL_indices])
        
        # 右左子樹 (X[2] <= 0.5)
        Root_RL_indices = Root_R_node.data[:, 2] <= 0.5
        Root_R_node.left = DecisionTreeNode(data=Root_R_node.data[Root_RL_indices])
        
        # 右右子樹 (X[2] > 0.5)
        Root_RR_node = DecisionTreeNode(feature=1, threshold=75.5, data=Root_R_node.data[~Root_RL_indices])
        Root_R_node.right = Root_RR_node
        
        # 右右左子樹 (X[1] <= 75.5)
        Root_RRL_indices = Root_RR_node.data[:, 1] <= 75.5
        Root_RR_node.left = DecisionTreeNode(data=Root_RR_node.data[Root_RRL_indices])
        
        # 右右右子樹 (X[1] > 75.5)
        Root_RR_node.right = DecisionTreeNode(data=Root_RR_node.data[~Root_RRL_indices])
        
        return root_node
    
                
    def predict(self, data):
        predictions = []
        for sample in data:
            node = self.root
            while node.left or node.right:
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(np.mean(node.data[:, 3]))
        return np.array(predictions)


df = pd.read_csv('HW6_roc.csv')
df["Outlook"] = df["Outlook"].replace(["sunny", "overcast", "rain"], [1, 2, 3])
df["Windy"] = df["Windy"].replace(["T", "F"], [1, 0])
df["Play"] = df["Play"].replace(["T", "F"], [1, 0])

data = df.values


dt = DecisionTree()
dt.fit(data)


scores = dt.predict(data)

score_unique = np.unique(scores)

fig, ax = plt.subplots(1, len(score_unique)+1, figsize=(25, 4))
tpr = list()
fpr = list()

for i in range(len(score_unique)):
    predictions = np.where(scores >= score_unique[i], 1, 0)
    tp = np.sum((predictions == 1) & (df["Play"] == 1))
    fp = np.sum((predictions == 1) & (df["Play"] == 0))
    fn = np.sum((predictions == 0) & (df["Play"] == 1))
    tn = np.sum((predictions == 0) & (df["Play"] == 0))
    tpr_ = tp / (tp + fn)
    fpr_ = fp / (tn + fp)
    tpr.append(tpr_)
    fpr.append(fpr_)
    confusion_matrix = np.array([[tn, fp], 
                                 [fn, tp]])
    img = ax[i].imshow(confusion_matrix, cmap='Blues')
    ax[i].text(0, 0, tn, ha='center', va='center', color='black')
    ax[i].text(0, 1, fn, ha='center', va='center', color='black')
    ax[i].text(1, 0, fp, ha='center', va='center', color='black')
    ax[i].text(1, 1, tp, ha='center', va='center', color='black')
    ax[i].set_title(f'Confusion Matrix (score:{score_unique[i]})')
    ax[i].set_xlabel('Predicted')
    ax[i].set_ylabel('Actual')
    ax[i].set_xticks([0, 1])
    ax[i].set_yticks([0, 1])
    plt.colorbar(img, ax=ax[i])
ax[-1].plot(fpr, tpr, marker='o')
ax[-1].set_title('ROC Curve')
ax[-1].set_xlabel('FPR')
ax[-1].set_ylabel('TPR')
ax[-1].set_xlim(0, 1)
ax[-1].set_ylim(0, 1)
ax[-1].grid()
plt.show()



fpr, tpr, thresholds = roc_curve(df["Play"], scores)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
disp.plot()
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid()
plt.show()
