import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class myNB_Gaussian_Classifier:
    
    def fit(self, df):
        self.male_stats = {'mean': df[df['Person'] == 1][['height', 'weight', 'foot size']].mean(),
                           'var': df[df['Person'] == 1][['height', 'weight', 'foot size']].var()}
        self.female_stats = {'mean': df[df['Person'] == 0][['height', 'weight', 'foot size']].mean(),
                             'var': df[df['Person'] == 0][['height', 'weight', 'foot size']].var()}
    
    def predict(self, X):
        P_male = 0.5
        P_female = 0.5
        for i, feature in enumerate(sample):
            P_male *= self.gaussian_distribution(feature, self.male_stats['mean'][i], self.male_stats['var'][i])
        for i, feature in enumerate(sample):
            P_female *= self.gaussian_distribution(feature, self.female_stats['mean'][i], self.female_stats['var'][i])
            
        if P_male >= P_female:
            return 'male'
        else: 
            return 'female'
            
    def gaussian_distribution(self, x, mean, variance):
        exponent = np.exp(-((x - mean) ** 2) / (2 * variance))
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent
    
    def plot_distributions(self):
        features = ['height', 'weight', 'foot size']
        for i, feature in enumerate(features):

            x_min = min(self.male_stats['mean'][i] - 4 * np.sqrt(self.male_stats['var'][i]),
                         self.female_stats['mean'][i] - 4 * np.sqrt(self.female_stats['var'][i]))
            x_max = max(self.male_stats['mean'][i] + 4 * np.sqrt(self.male_stats['var'][i]),
                         self.female_stats['mean'][i] + 4 * np.sqrt(self.female_stats['var'][i]))
            x = np.linspace(x_min, x_max, 100)
            
            # Male and Female PDFs
            male_distribution = self.gaussian_distribution(x, self.male_stats['mean'][i], self.male_stats['var'][i])
            female_distribution = self.gaussian_distribution(x, self.female_stats['mean'][i], self.female_stats['var'][i])
            
            plt.figure(figsize=(10, 5))
            plt.plot(x, male_distribution, label='Male Distribution', color='blue')
            plt.plot(x, female_distribution, label='Female Distribution', color='red')
            plt.title(f'Probability Distribution of {feature}', fontsize = 20)
            plt.xlabel(feature, fontsize = 16)
            plt.ylabel('Probability Density', fontsize = 16)
            plt.legend()
            plt.grid()
            plt.show()

# 示例训练数据
data = {
    'height': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
    'weight': [180, 190, 170, 165, 100, 150, 130, 150],
    'foot size': [12, 11, 12, 10, 6, 8, 7, 9],
    'Person': ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female']
}

# 创建DataFrame
df_train = pd.DataFrame(data)

# 将性别转换为数值
df_train['Person'] = df_train['Person'].map({'male': 1, 'female': 0})

model = myNB_Gaussian_Classifier()
model.fit(df_train)

model.plot_distributions()

# 示例分类
sample = (6, 130, 8)  # 待分类样本（身高，体重，鞋码）
prediction = model.predict(sample)
print(f'{sample[0]} feet, {sample[1]} lbs, {sample[2]} inches is: {prediction}')
