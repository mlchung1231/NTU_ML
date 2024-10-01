import numpy as np


def P_y_given_x(y,x):
    if y == 1:
        if abs(x) > 0.5:
            return 0.9
        else:
            return 0.1
    elif y == -1:
        if abs(x) > 0.5:
            return 0.1
        else:
            return 0.9
    else:
        return None
        
def u_a(x):
    return 0.5

def u_b(x):
    return abs(x)

def h(x, x1, x2):
    predict = np.sign((x - x1) * (x - x2))
    
    return predict


def L_D(x_domain, x1_doamin, x2_domain, dx):
    error_min_a = 10000
    error_min_b = 10000
    for x1 in x1_domain:
        for x2 in x2_domain: 
            total_error_a = 0
            total_error_b = 0
            for x in x_domain:
                if x == -0.5 or x == 0.5:
                    continue
                if h(x, x1, x2) == 0:
                    error_a = 0
                    error_b = 0
                elif  h(x, x1, x2) == 1:
                    error_a = P_y_given_x(-1, x) * u_a(x) * dx
                    error_b = P_y_given_x(-1, x) * u_b(x) * dx
                else:
                    error_a = P_y_given_x(1, x) * u_a(x) * dx
                    error_b = P_y_given_x(1, x) * u_b(x) * dx
                    
                total_error_a += error_a
                total_error_b += error_b
                
            if total_error_a <= error_min_a:
                error_min_a = total_error_a
                x1_a = x1
                x2_a = x2
            if total_error_b <= error_min_b:
                error_min_b = total_error_b
                x1_b = x1
                x2_b = x2
            
            
    return error_min_a, x1_a, x2_a, error_min_b, x1_b, x2_b
        

dx = 0.01
"""
x_domain = [-1 + i * dx for i in range(201)]
x1_domain = [-0.99 + i * dx for i in range(199)]
x2_domain = [-0.99 + i * dx for i in range(199)]
"""
x_domain = np.arange(-1.0,1.01, 0.01 )
x1_domain = np.arange(-0.99,1.0, 0.01 )
x2_domain = np.arange(-0.99,1.0, 0.01 )


error_a, x1_a, x2_a, error_b, x1_b, x2_b = L_D(x_domain, x1_domain, x2_domain, dx)

print("(a) If probability distribution x is uniform:")
print(f"    minimum LD(h) = {error_a:.3f}")
print(f"    x1 = {x1_a:.3f}")
print(f"    x2 = {x2_a:.3f}\n")
print("(b) If probability distribution x = |x|:")
print(f"    minimum LD(h) = {error_b:.3f}")
print(f"    x1 = {x1_b:.3f}")
print(f"    x2 = {x2_b:.3f}")




