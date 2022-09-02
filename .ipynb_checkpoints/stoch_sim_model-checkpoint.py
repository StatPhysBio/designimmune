### Stochastic model dynamics ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import special
from math import erf
import numba as nb

### (1) Define simulation parameters
# Define simulation parameters

# antigens
a1_0 = 100_000
a2 = 50_000_000
a1_max = 0.4*a2
b_a1 = 10
t1 = 10
d_a1 = 3.8*10**(-4) # proxy for virulence
T_a1 = 10

t2 = 0.5 # dissociation time
n, m = 4, 2

# cells
N0 = 1000
Treg0 = 10000 # initial Tregs
g_NE_max = 0.36
g_NM_max = 0.12
b_E_max = 2.0
g_ME_max = 0 #4.0
g_EM_max = 0.036

k_E = 10**(-11)*(6.0221408*10**23)*50*10**(-6)
k_M = k_E
d_E_max = 2.5
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
init_state = np.array([a1_0, N0, 0, 0, 0, Treg0]) #a1, N, E, cM, eM, T

# activation threshold
mu_theta = 5_000_000
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
def c1_ss(t1,a1, E, T):
    
    return (E*b_c1*p_A(t1,a1) + I_c1)/(T*f_T + 1/tau_c)

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

    return g_ME_max*(hl_u(c1_ss(t1,a1, E, T),k_M))

@nb.njit
def g_EM(T):
    
    return g_EM_max

@nb.njit
def d_E(t1,a1, E, T):
    
    return d_E_max*(1-hl_u(c1_ss(t1,a1, E, T), k_E))

@nb.njit
def d_T(T):
    
    return 0 #(1-hl_u(c1_ss(a1,E,T),k_T))/tau_T

# define dynamics
@nb.njit
def state_dyn(t, z, b_a1, t1, d_a1, T_a1):
    
    
    a1, N, E, cM, eM, T = z
    
    #P = P1(a1,E)[0] # set to 0 for noiseless, 1 for noise
    #P_vec[np.argmax(P_vec == -999)] = P # (1) uncomment to debug noise
    
    return np.asarray([a1*b_a1*(1- a1/a1_max - t/T_a1) - d_a1*a1*E,\
            -(g_NE(T)+g_NM(T))*N*p_A(t1,a1),\
            g_NE(T)*N*p_A(t1,a1) + cM*g_ME(t1,a1,E,T) +  E*(b_E(T) - g_EM(T) - d_E(t1,a1,E,T)), \
            g_NM(T)*N*p_A(t1,a1) - cM*g_ME(t1,a1,E,T), \
            E*g_EM(T),\
            0*T*(b_T(T) - d_T(T))])

### (3) Code to run individual simulation
# Run single simulation and plot outputs
def stoch_sim(b_a1 = 2, t1 = 10, d_a1 = 3.8*10**(-4), T_a1 = 10, duration = 20, steps = 10**4):
    
    dt = duration/steps
    ts = np.linspace(0, duration, steps + 1)
    
    states = solve_ivp(state_dyn, [0, duration], init_state,
                    dense_output=True, args=[b_a1, t1, d_a1, T_a1]).sol(ts).T

    a1, N, E, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]
    
    return a1, N, E, cM, eM, T, ts
