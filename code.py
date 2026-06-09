import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

#constants
c = 2.71099e-7
m = 1e-6 
d = m / (3 * np.pi) 

#radius
r = np.linspace(50, 500, 70)

# Predicted data (s_pred) 
s_pred = c * (r/213)**(-0.39) * np.exp(-(r/213)**1.61)
s_pred = np.maximum(s_pred, 1e-20) 

# Model Function
def compute_s(r_val, k, n):
    k = np.maximum(k, 1e-12)
    n = np.maximum(n, 1e-12)
    
    # model: d / (k * r^n) * (1 + sqrt(50/r))
    s = (d / (k * (r_val ** n))) * (1 + np.sqrt(50 / r_val))
    
    return np.maximum(s, 1e-20)


def objective(params):
    k, n = params
    s_model = compute_s(r, k, n)
    
    # Minimize the sum of squared differences of the BASE 10 logarithms
    log_s_pred = np.log10(s_pred)
    log_s_model = np.log10(s_model)
    
    return np.sum((log_s_pred - log_s_model)**2)


p0 = [7e-5, 1] #initial guesses


bounds = [(1e-9, 1e-3), (1e-9, 3)] 

result = minimize(
    objective, 
    p0, 
    bounds=bounds, 
    method='SLSQP',
    options={'maxiter': 1000}
)

k_opt, n_opt = result.x

# --- Results and Plotting ---
if result.success:
    print(f"Optimization successful with log objective (SLSQP).")
    print(f"Optimized k = {k_opt:.6e}")
    print(f"Optimized n = {n_opt:.6f}")
else:
    print(f"Optimization failed. Status: {result.message}")
    k_opt, n_opt = p0 
    
# Compute the optimized s values
s_opt = compute_s(r, k_opt, n_opt)


plt.figure(figsize=(10, 6))
plt.plot(r, s_pred, 'o-', label='s ')
plt.plot(r, s_opt, 'o--', label=f's_pred ( k={k_opt:.2e}, n={n_opt:.2f})')
plt.xlabel('r', fontsize=12)
plt.ylabel('s', fontsize=12)
plt.title(' Curve Fitting: s_pred vs. s(k, n) ', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.show()

                                
    


