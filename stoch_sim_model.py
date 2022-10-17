### Stochastic model dynamics ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import special
from math import erf
import numba as nb
import os as os

### (1) Define simulation parameters
# Define simulation parameters

# antigens
a1_0 = 100
a2 = 50_000_000
a1_max = 0.4*a2
b_a1 = 10
t1 = 10
d_a1 = 3.8*10**(-4) # proxy for virulence
T_a1 = 2

t2 = 0.5 # dissociation time
n, m = 4, 2

# cells
N0 = 10000
Treg0 = 10000 # initial Tregs
g_NE_max = 0 #0.36
g_NM_max = 4 #0.12
b_E_max = 2.0
g_ME_max = 6.0
g_EM_max = 0.07

k_E = 10**(-11)*(6.0221408*10**23)*50*10**(-6)
k_M = k_E
d_E_max = 1
g_T_max = 4
d_T_max = 6

# cytokines
b_c1 = 1000*3600*24
b_c2 = 10 #b_c1/100
f_T = 3*10**(7)*3000*3600*24/((6.0221408*10**23)*50*10**(-6))
k_c2_a = 10**(-14)*(6.0221408*10**23)*50*10**(-6)
tau_c = 5/24
I_c1 = 0
I_c2 = 0

# set initial state
init_state = np.array([N0, 0, 0, 0, Treg0]) # N, E, cM, eM, T

# activation threshold
mu_theta = 100_00 # set to point at which activation probabity equals antigen frequency
sigma_theta = np.log((mu_theta*t1**n +t2**n)/(mu_theta*t1**m + t2**m))/10

### (2) Define functions for simulations
# Define functions for simulations
@nb.vectorize([nb.float64(nb.float64)])
def verf(x):
    return erf(x)

@nb.njit
def hl_u(x,k):
    return (x**1)/(k**1 + x**1)

@nb.njit
def a_stim(t1,a1):
    return (a1/a2*t1**n +t2**n)/(a1/a2*t1**m + t2**m)

@nb.njit
def p_A(t1,a1):
    
    return 1/2 + verf(np.log(a_stim(t1,a1)/a_stim(t1,mu_theta))/np.sqrt(2*sigma_theta**2))/2

@nb.njit
def c2_ss(a1, E):
    
    return E*b_c2*tau_c

@nb.njit
def c1_ss(t1,a1, E, T, b_c1_pop = b_c1):
    
    return (E*b_c1_pop*p_A(t1,a1) + I_c1)/(T*f_T + 1/tau_c)

@nb.njit
def g_NE(T):
    
    return g_NE_max

@nb.njit
def g_NM(T):
    
    return g_NM_max

@nb.njit
def b_E(T):
    
    return b_E_max

@nb.njit
def b_T(T):
    
    return g_T_max

@nb.njit
def g_ME(t1,a1,E,T):

    return g_ME_max*p_A(t1,a1)

@nb.njit
def g_EM(t1,a1,E,T):
    
    return g_EM_max*(1-p_A(t1,a1))

@nb.njit
def d_E(t1,a1, E, T, b_E_pop = b_E_max, b_c1_pop = b_c1):
    
    return (b_E_pop + d_E_max)*(1-hl_u(c1_ss(t1,a1, E, T, b_c1_pop), k_E))

@nb.njit
def d_T(T):
    
    return 0 #(1-hl_u(c1_ss(a1,E,T),k_T))/tau_T

# define dynamics
@nb.njit
def state_dyn(t, z, a1_0, b_a1, t1, d_a1, T_a1):
    
    
    a1, N, E, cM, eM, T = z
    
    return np.asarray([a1*b_a1*(1- a1/a1_max)*(1 - (a1*b_a1*(1 - a1/a1_max) - d_a1*a1*E <= 0)*(a1 <= mu_theta))*(a1 >= a1_0) - d_a1*a1*E,\
            -(g_NE(T)+g_NM(T))*N*p_A(t1,a1),\
            g_NE(T)*N*p_A(t1,a1) + cM*g_ME(t1,a1,E,T) +  E*(b_E(T) - g_EM(t1,a1,E,T) - d_E(t1,a1,E,T)), \
            g_NM(T)*N*p_A(t1,a1) - cM*g_ME(t1,a1,E,T), \
            E*g_EM(t1,a1,E,T),\
            0*T*(b_T(T) - d_T(T))])

@nb.njit
def pop_state_dyn(t, z, a1_0, b_a1, t1, d_a1, b_E_pop, g_ME_pop, b_c1_pop, T_a1 = T_a1):
    
    
    a1, N, E, cM, eM, T = z
    
    return np.asarray([a1*b_a1*(1- a1/a1_max)*(1 - (a1*b_a1*(1 - a1/a1_max) - d_a1*a1*E <= 0)*(a1 <= mu_theta))*(a1 >= a1_0) - d_a1*a1*E,\
            -(g_NE(T)+g_NM(T))*N*p_A(t1,a1),\
            g_NE(T)*N*p_A(t1,a1) + cM*g_ME_pop*p_A(t1, a1) +  E*(b_E_pop - g_EM(t1,a1,E,T) - d_E(t1, a1, E, T, b_E_pop, b_c1_pop)), \
            g_NM(T)*N*p_A(t1, a1) - cM*g_ME_pop*p_A(t1, a1), \
            E*g_EM(t1, a1, E, T),\
            0*T*(b_T(T) - d_T(T))])

### (3) Code to run individual simulation
# Run single simulation and plot outputs
duration = 20
steps = 10**4

def stoch_sim(a1_0 = a1_0, b_a1 = b_a1, t1 = t1, d_a1 = d_a1, T_a1 = T_a1, noise_model = "pop", noise_cv = [0.0, 0.0, 0.0]):

    
    dt = duration/steps
    ts = np.linspace(0, duration, steps + 1)
    
    if noise_model == "pop":
        b_E_pop = np.random.lognormal(mean = np.log(b_E_max), sigma = noise_cv[0]*np.abs(np.log(b_E_max)))
        g_ME_pop = np.random.lognormal(mean = np.log(g_ME_max), sigma = noise_cv[1]*np.abs(np.log(g_ME_max)))
        b_c1_pop = np.random.lognormal(mean = np.log(b_c1), sigma = noise_cv[2]*np.abs(np.log(b_c1)))
        
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([a1_0], init_state), axis = None),
                    dense_output=True, args=[a1_0, b_a1, t1, d_a1, b_E_pop, g_ME_pop, b_c1_pop, T_a1,]).sol(ts).T
    else:
        states = solve_ivp(state_dyn, [0, duration], np.concatenate(([a1_0],init_state), axis = None),
                    dense_output=True, args=[a1_0, b_a1, t1, d_a1, T_a1]).sol(ts).T

    a1, N, E, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]
    
    return a1, N, E, cM, eM, T, ts

### (4) Parallelize simulation runs
def sum_sim(a1_0 = a1_0, b_a1 = b_a1, t1 = t1, d_a1 = d_a1, T_a1 = T_a1, noise_cv = [0.1,0.0,0.0]):
    # compute state and costate dynamics
    dt = duration/steps
    a1, N, E, cM, eM, T, ts = stoch_sim(a1_0, b_a1, t1, d_a1, T_a1, noise_cv = noise_cv)
    run_data = [a1_0, b_a1, t1, d_a1, np.sum(np.log(a1+1)*dt), np.argmax(a1)*dt, np.max(E)/(np.argmax(E)*dt), \
                np.argmax(E)*dt, (eM[-1]+cM[-1])/np.max(E), np.sum(np.log(E+1)*dt)]
    return run_data

stat_names = [r"$a_1^0$", r"$b_{a_1}$",r"$t_1$",r"$d_{a_1}$",r"$\int \log(a_1+1) dt$",\
                             r"$T_{a_1}^{max}$", r"$\frac{E^{max}}{T_{E}^{max}}$",r"$T_{E}^{max}$",\
                             r"$\frac{(cM + eM)^\infty}{E^{max}}$",r"$\int \log(E+1) dt$"]
stat_names_for_df = ['a1_0','b_a1', 't_1','d_a1',\
                     'int_loga1','T_a1_max', 'E_max_T_E_max',\
                     'T_E_max','mem_frac','int_logE']

### (5) define basic mutual information function
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bin_num = 500):
    # bx = np.histogram_bin_edges(x, bins="sqrt", range=None, weights=None)
    # by = np.histogram_bin_edges(y, bins="sqrt", range=None, weights=None)
    
    c_xy = np.histogram2d(x, y, bin_num)[0]
    mi_raw = mutual_info_score(None, None, contingency=c_xy)/np.log(2)
    
    # MI correction by shuffling data
    c_xy_shuffle = np.histogram2d(x, y[np.random.permutation(y.shape[0])], bin_num)[0]
    mi_correction = mutual_info_score(None, None, contingency=c_xy_shuffle)/np.log(2)
    
    return mi_raw - mi_correction