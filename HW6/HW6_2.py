import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sympy as sp
from sympy import plot_implicit, symbols, Eq
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay

class myGNB_Classifier:
    
    def fit(self, data):
        self.parameter = [[], []]
        for i in range(2):
            class_data = data[data[:, 2] == i][:, 0:2]
            self.parameter[i] = [np.mean(class_data, axis=0),
                                 np.var(class_data, axis=0),
                                 class_data.shape[0] / data.shape[0]]
        self.X = data[:,0:2]
        self.Y = data[:,2]
        self.x_max = self.X.max(axis=0)
        self.x_min = self.X.min(axis=0)
        
    def print_parameter(self):
        print(["parabola", "hyperbola", "ellipse"][i])
        print("µ   [µ1-, µ2-, µ1+, µ2+]:", self.parameter[0][0], self.parameter[1][0])
        print("σ^2 [σ1-, σ2-, σ1+, σ2+]:", self.parameter[0][1], self.parameter[1][1])
        print("p   [p-, p+]:", [self.parameter[0][2], self.parameter[1][2]])
        print()
    
    def plot_DB(self, DB):
        DB_substituted = DB.subs({
            avg[0]: self.parameter[0][0][0],
            avg[1]: self.parameter[0][0][1],
            avg[2]: self.parameter[1][0][0],
            avg[3]: self.parameter[1][0][1],
            var[0]: self.parameter[0][1][0],
            var[1]: self.parameter[0][1][1],
            var[2]: self.parameter[1][1][0],
            var[3]: self.parameter[1][1][1],
            p[0]: self.parameter[0][2],
            p[1]: self.parameter[1][2]
        })

        DB_eq = Eq(DB_substituted, 0)
        
        x1, x2 = symbols('x1 x2')
        plot = plot_implicit(DB_eq, (x1, self.x_min[0], self.x_max[0]), 
                               (x2, self.x_min[1], self.x_max[1]), show=False)
        return plot
    
    
    def plot_DB_withData(self, plot):
        point = np.array(plot[0].get_points()[0])
        q = lambda x:x.mid
        px = list(map(q, point[:,0]))
        py = list(map(q, point[:,1]))
        plt.scatter(self.X[:,0][self.Y==1], self.X[:,1][self.Y==1], color='r')
        plt.scatter(self.X[:,0][self.Y==0], self.X[:,1][self.Y==0], color='b')
        plt.scatter(px, py, color='g', marker='.', s=0.3)
        plt.grid()
        plt.show()

def sklean_ans():
    model = list()
    x1_min = list()
    x1_max = list()
    x2_min = list()
    x2_max = list()
    X_ = list()
    y_ = list()
    
    for i in range(3):
        model.append(GaussianNB())
    
        sc = StandardScaler()
        X = np.array([data[i]['x1'], data[i]['x2']]).transpose()
        X = sc.fit_transform(X)
        
        x1_max.append(np.max(X[:,0]))
        x2_max.append(np.max(X[:,1]))
        x1_min.append(np.min(X[:,0]))
        x2_min.append(np.min(X[:,1]))
        
        y = np.array(data[i]['y'].values)
        
        X_.append(X)
        y_.append(y)
        
        model[i].fit(X, y)
        
        print(["parabola", "hyperbola", "ellipse"][i])
        print('µ   [µ1-, µ2-, µ1+, µ2+]:', model[i].theta_[0], model[i].theta_[1])
        print('σ^2 [σ1-, σ2-, σ1+, σ2+]:', model[i].var_[0], model[i].var_[1])
        print("p   [p- , p+ ]:", model[i].class_prior_)
        print()
        
    for i in range(3):
        feature_1, feature_2 = np.meshgrid(
            np.linspace(x1_min[i], x1_max[i], num=300),
            np.linspace(x2_min[i], x2_max[i], num=300))
        grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
        y_pred = np.reshape(model[i].predict(grid), feature_1.shape)
        display = DecisionBoundaryDisplay(xx0=feature_1, xx1=feature_2, response=y_pred)
    
        display.plot(cmap="twilight")
        display.ax_.scatter(X_[i][:,0], X_[i][:,1], c=y_[i], cmap="bwr", alpha=0.5)
        plt.show()


data_par = pd.read_csv("data_parabola.csv", names=['x1', 'x2', 'y'], header=None, dtype='float')
data_hyp = pd.read_csv("data_hyperbola.csv", names=['x1', 'x2', 'y'], header=None, dtype='float')
data_ell = pd.read_csv("data_ellipse.csv", names=['x1', 'x2', 'y'], header=None, dtype='float')
data = [data_par, data_hyp, data_ell]


fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i in range(3):
    ax[i].scatter(data[i]['x1'][data[i]['y']==1], data[i]['x2'][data[i]['y']==1], color='r')
    ax[i].scatter(data[i]['x1'][data[i]['y']==0], data[i]['x2'][data[i]['y']==0], color='b')

ax[0].set_title("parabola", fontsize = 20)
ax[1].set_title("hyperbola", fontsize = 20)
ax[2].set_title("ellipse", fontsize = 20)
plt.tight_layout()
plt.show()


data_scaled = []
for i in range(3):
    X = np.array([data[i]['x1'], data[i]['x2']]).transpose()
    Y = np.array([data[i]['y']]).transpose()
    scaler = StandardScaler().fit(X)
    scaler_x = scaler.transform(X)
    scaler_xy = np.append(scaler_x, Y, axis=1)
    data_scaled.append(scaler_xy)


model = []
for i in range(3):
    model.append(myGNB_Classifier())
    model[i].fit(data_scaled[i])
    model[i].print_parameter()


avg = symbols('µ_{1}- µ_{2}- µ_{1}+ µ_{2}+')
var = symbols('σ_{1}^{2}- σ_{2}^{2}- σ_{1}^{2}+ σ_{2}^{2}+')
p = symbols('p- p+')
x1, x2 = symbols('x1 x2')
DB = ((0.5*sp.ln((2*sp.pi*var[0])*(2*sp.pi*var[1])/((2*sp.pi*var[2])*(2*sp.pi*var[3]))))
       +(sp.ln(p[1]/(p[0])))
       -0.5*(-((x2-avg[1])/sp.sqrt(var[1]))+((x2-avg[3])/sp.sqrt(var[3])))*(((x2-avg[1])/sp.sqrt(var[1]))+((x2-avg[3])/sp.sqrt(var[3])))
       -0.5*(-((x1-avg[0])/sp.sqrt(var[0]))+((x1-avg[2])/sp.sqrt(var[2])))*(((x1-avg[0])/sp.sqrt(var[0]))+((x1-avg[2])/sp.sqrt(var[2]))))
DB

plot_DB = []
for i in range(3):
    print(["parabola", "hyperbola", "ellipse"][i])
    plot = model[i].plot_DB(DB)
    plot_DB.append(plot)
    plot.show()  
    model[i].plot_DB_withData(plot)
    

sklean_ans()
    

    