import numpy as np
import matplotlib.pyplot as plt

n = 100
np.random.seed(0)
a = np.random.uniform(-1,1,n)
b = np.random.uniform(-1,1,n)


def f_i(x, y, a_i, b_i):
    return (x - a_i)**2 + (y - b_i)**2

def f(x, y, a, b):
    return np.mean([(f_i(x, y, a[i], b[i])) for i in range(n)])


def grad_f_i(x, y, a_i, b_i):
    grad_x = 2 * (x - a_i)
    grad_y = 2 * (y - b_i)
    return grad_x, grad_y

def grad_f(x, y, a, b, batch):
    grad_x = np.mean([grad_f_i(x, y, a[i], b[i])[0] for i in range(batch)])
    grad_y = np.mean([grad_f_i(x, y, a[i], b[i])[1] for i in range(batch)])
    return grad_x, grad_y


def SGD(eta, a, b, iteration, batch, decay=False):
    x, y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
    loss = []
    
    for t in range(1,iteration+1):
        i = np.random.randint(0, n)
        grad_x, grad_y = grad_f_i(x, y, a[i], b[i])
        
        if decay:
            eta_t = eta / t
        else:
            eta_t = eta
        
        x -= eta_t * grad_x
        y -= eta_t * grad_y
        
        dist2 = f(x, y, a, b)
        loss.append(dist2)
    
    return loss


eta = 0.5  
batch = 4
iteration = [10, 100, 1000]


loss_fix_eta = []
loss_decay_eta = []

for i in range(3):
    for _ in range(5):
        loss_fix_eta.append(SGD(eta, a, b, iteration[i], batch, decay=False))
        loss_decay_eta.append(SGD(eta, a, b, iteration[i], batch, decay=True))
    
    
    plot_loss_fix = np.mean(loss_fix_eta, axis=0)
    plot_loss_decay = np.mean(loss_decay_eta, axis=0)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(plot_loss_fix)
    plt.title('SGD with Fixed Learning Rate\n'+'iteration='+str(iteration[i])+', batch='+str(batch)+'\ndistance='+str(plot_loss_fix[-1]),fontsize=16)
    plt.xlabel('Iterations',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.grid()
    
    
    plt.subplot(1, 2, 2)
    plt.plot(plot_loss_decay)
    plt.title('SGD with Decaying Learning Rate\n'+'iteration='+str(iteration[i])+', batch='+str(batch)+'\ndistance='+str(plot_loss_decay[-1]),fontsize=16)
    plt.xlabel('Iterations',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    
    plt.grid()
    plt.show()
    
    loss_fix_eta.clear()
    loss_decay_eta.clear()
