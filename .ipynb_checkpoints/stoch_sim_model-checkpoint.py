### Stochastic model dynamics ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import special
from math import erf
import numba as nb
import os as os
import pandas as pd

### (1) Define simulation parameters
# Define simulation parameters

# antigens
I_0 = 10_000 # initial detectable levelof infected cells
a2 = 20_000_000
b_I = 0.00003
t1 = 10
d_IE = 3.8*10**(-4) # proxy for virulence
d_I = 1*10**(-1)
T_I = 2
S0 = 10_000_000 #susceptible cells

t2 = 2 # dissociation time
n, m = 4, 2

# cells
N0 = 100
Treg0 = 10000 # initial Tregs
g_NE_max = 0.50 #0.36
g_NM_max = 1 #0.12
b_E_max = 2.0
g_ME_max = 6.0
g_EM_max = 0.07

E_min = 10 # minimum detectable cell counts

d_E_max = b_E_max
b_T_max = 4
d_T_max = 6

# cytokines
l = 2
b_c1 = 1000*3600*24
b_c2 = 10 #b_c1/100
f_T = 3*10**(7)*3000*3600*24/((6.0221408*10**23)*50*10**(-6))
k_c2_a = 10**(-14)*(6.0221408*10**23)*50*10**(-6)
tau_c = 0.015*24
I_c1 = 0
I_c2 = 0

# cellular cytokine thresholds
k_E = 10**(-11)*(6.0221408*10**23)*50*10**(-6)
k_M = k_E

# set initial state
init_state = np.array([N0, 0, 0, 0, Treg0]) # N, E, cM, eM, T

# activation threshold
mean_theta = 50_000_000 # set to point at which activation probabity equals antigen frequency
cv_theta = 0.5 #np.log((mu_theta*t1**n +t2**n)/(mu_theta*t1**m + t2**m))/10

# hyper parameters
alpha = 0.5 # antigen-cyokine weighting
psi_N_I, psi_cM_I, psi_E_I, psi_E_c = 0.5, 0.5, 0.1, 0.1 # decision to upregulate or downregulate based on stimulus

### (2) Define functions for simulations
# Define functions for simulations
@nb.vectorize([nb.float64(nb.float64)])
def verf(x):
    return erf(x)

@nb.njit
def hl_u(x,k):
    return (x**l)/(k**l + x**l)

@nb.njit
def a_1(t1,I):
    return 20*I

@nb.njit
def a_stim(t1,I):
    return (a_1(t1,I)/a2*t1**n +t2**n)/(a_1(t1,I)/a2*t1**m + t2**m)

@nb.njit
def p_A(t1,I):
    
    mu = np.log(a_stim(4*t2, mean_theta)/np.sqrt(1 + cv_theta**2))
    sigma = np.sqrt(np.log(1 + cv_theta**2))
    
    return 1/2 + verf((np.log(a_stim(t1,I))-mu)/np.sqrt(2*sigma**2))/2

@nb.njit
def c2_ss(I, E):
    
    return E*b_c2*tau_c

@nb.njit
def c1_ss(t1,I, E, T, b_c1_pop = b_c1):
    
    return (E*b_c1_pop*p_A(t1,I))/(T*f_T*(1-p_A(t1,I)) + 1/tau_c)

@nb.njit
def g_N(t1,I,E,T, max_val = g_NE_max):
    
    return max_val*p_A(t1,I)

@nb.njit
def g_NE(t1,I,E,T, max_val = g_NE_max):
    
    return max_val*p_A(t1,I)

@nb.njit
def g_NM(t1,I,E,T, max_val = g_NM_max):
    
    return max_val*p_A(t1,I)

@nb.njit
def b_E(t1,I, E, T, max_val = b_E_max, b_c1_pop = b_c1):
    
    return max_val*hl_u(c1_ss(t1,I, E, T, b_c1_pop), k_E)

@nb.njit
def b_T(T):
    
    return b_T_max

@nb.njit
def b_cM(t1,I,E,T, max_val = g_ME_max):

    return max_val*p_A(t1,I)

@nb.njit
def g_ME(t1,I,E,T, max_val = g_ME_max):

    return max_val*p_A(t1,I)

@nb.njit
def g_EM(t1,I,E,T, max_val = g_EM_max):
    
    return max_val*(1-p_A(t1,I))

@nb.njit
def d_E(t1,I, E, T, b_E_pop = b_E_max, b_c1_pop = b_c1):
    
    return (b_E_pop + d_E_max)*(1-hl_u(c1_ss(t1,I, E, T, b_c1_pop), k_E))

@nb.njit
def d_T(T):
    
    return 0 #(1-hl_u(c1_ss(I,E,T),k_T))/tau_T

# define dynamics
@nb.njit
def state_dyn(t, z, I_0, b_I, t1, d_IE, T_I):
    
    
    I, N, E, cM, eM, T = z
    
    return np.asarray([I*b_I*(S0-I)*(1 - (I*b_I*(S0-I) - d_IE*I*E <= 0)*(a_1(t1,I) <= mu_theta)) - d_IE*I*E,\
            -(g_NE(t1,I,E,T)+g_NM(t1,I,E,T))*N,\
            g_NE(t1,I,E,T)*N + cM*g_ME(t1,I,E,T) +  E*(b_E(t1,I,E,T) - g_EM(t1,I,E,T) - d_E(t1,I,E,T)), \
            g_NM(t1,I,E,T)*N - cM*g_ME(t1,I,E,T), \
            E*g_EM(t1,I,E,T),\
            0*T*(b_T(T) - d_T(T))])

@nb.njit
def pop_state_dyn(t, z, I_0, b_I, t1, d_IE, b_E_pop, b_c1_pop, 
                  g_NE_pop, g_NM_pop, g_ME_pop, g_EM_pop, T_I = T_I):
    
    
    I, N, E, cM, eM, T = z
    
    return np.asarray([(I >= 1)*(np.exp(-t/T_I)*I*b_I*(S0-I) - d_IE*I*E - d_I*I),\
            -(g_NE(t1,I,E,T, g_NE_pop)+g_NM(t1,I,E,T, g_NM_pop))*N,\
            g_NE(t1,I,E,T, g_NE_pop)*N + cM*g_ME(t1,I,E,T, g_ME_pop) +  E*(b_E_pop - g_EM(t1,I,E,T, g_EM_pop) - d_E(t1, I, E, T, b_E_pop, b_c1_pop)), \
            g_NM(t1,I,E,T, g_NM_pop)*N - cM*g_ME(t1,I,E,T, g_ME_pop), \
            E*g_EM(t1, I, E, T, g_EM_pop),\
            0*T*(b_T(T) - d_T(T))])

@nb.njit
def sec_pop_state_dyn(t, z, I_0, b_I, t1, d_IE, b_E_pop, b_c1_pop, 
                  g_NE_pop, g_NM_pop, g_ME_pop, g_EM_pop, T_I = T_I):
    
    
    I, N, E, pM, cM, eM, T = z
    
    return np.asarray([(I >= 1)*(np.exp(-t/T_I)*I*b_I*(S0-I) - d_IE*I*E - d_I*I),\
            -(g_NE(t1,I,E,T, g_NE_pop)+g_NM(t1,I,E,T, g_NM_pop))*N,\
            g_NE(t1,I,E,T, g_NE_pop)*N + cM*g_ME(t1,I,E,T, g_ME_pop) + g_ME(t1,I,E,T, g_ME_pop)*pM +  E*(b_E_pop - g_EM(t1,I,E,T, g_EM_pop) - d_E(t1, I, E, T, b_E_pop, b_c1_pop)), \
            -g_ME(t1,I,E,T, g_ME_pop)*pM,\
            g_NM(t1,I,E,T, g_NM_pop)*N - cM*g_ME(t1,I,E,T, g_ME_pop), \
            E*g_EM(t1, I, E, T, g_EM_pop),\
            0*T*(b_T(T) - d_T(T))])

### (3) Code to run individual simulation
# Run single simulation and plot outputs
duration = 20
steps = 10**4

## ODE-based model
def stoch_sim(I_0 = I_0, b_I = b_I, t1 = t1, d_IE = d_IE, T_I = T_I,
              rates = [b_E_max, b_c1, g_NE_max, g_NM_max, g_ME_max, g_EM_max],
              noise_model = "pop", noise_cv = [0.5,0,0,0,0,0], infection = "prim"):

    
    dt = duration/steps
    ts = np.linspace(0, duration, steps + 1)
    
    if noise_model == "pop":
        b_E_pop = np.minimum((rates[0] > 0)*np.random.lognormal(mean = np.log(rates[0]/np.sqrt(1 + noise_cv[0]**2) + (rates[0] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[0]**2))), 10*rates[0])
        
        b_c1_pop = np.minimum((rates[1] > 0)*np.random.lognormal(mean = np.log(rates[1]/np.sqrt(1 + noise_cv[1]**2) + (rates[1] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[1]**2))), 10*rates[1])
        
        g_NE_pop = np.minimum((rates[2] > 0)*np.random.lognormal(mean = np.log(rates[2]/np.sqrt(1 + noise_cv[2]**2) + (rates[2] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[2]**2))), 10*rates[2])
        
        g_NM_pop = np.minimum((rates[3] > 0)*np.random.lognormal(mean = np.log(rates[3]/np.sqrt(1 + noise_cv[3]**2) + (rates[3] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[3]**2))), 10*rates[3])
        
        g_ME_pop = np.minimum((rates[4] > 0)*np.random.lognormal(mean = np.log(rates[4]/np.sqrt(1 + noise_cv[4]**2) + (rates[4] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[4]**2))), 10*rates[4])
        
        g_EM_pop = np.minimum((rates[5] > 0)*np.random.lognormal(mean = np.log(rates[5]/np.sqrt(1 + noise_cv[5]**2) + (rates[5] == 0)), sigma = np.sqrt(np.log(1+ noise_cv[5]**2))), 10*rates[5])

        
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([I_0], init_state), axis = None),
                    dense_output=True, args=[I_0, b_I, t1, d_IE, b_E_pop, b_c1_pop, g_NE_pop, g_NM_pop, g_ME_pop, g_EM_pop, T_I]).sol(ts).T
    else:
        states = solve_ivp(state_dyn, [0, duration], np.concatenate(([I_0],init_state), axis = None),
                    dense_output=True, args=[I_0, b_I, t1, d_IE, T_I]).sol(ts).T

    I, N, E, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]
    
    if infection == "sec" and noise_model == "pop":
        states = solve_ivp(sec_pop_state_dyn, [0, duration], np.concatenate(([I_0], np.array([N[-1], 0,cM[-1] + eM[-1], 0, 0, Treg0])), axis = None),
                    dense_output=True, args=[I_0, b_I, t1, d_IE, b_E_pop, b_c1_pop, g_NE_pop, g_NM_pop, g_ME_pop, g_EM_pop, T_I]).sol(ts).T
        I, N, E, pM, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5], states[:,6]
    else:
        pM = 0
    
    return np.array([b_E_pop, b_c1_pop, g_NE_pop, g_NM_pop, g_ME_pop, g_EM_pop]), I, N, E, cM, eM + pM, T, ts

## agent-based simulation with tau leaping
def agent_stoch_sim(I_0 = I_0, b_I = b_I, t1 = t1, d_IE = d_IE, T_I = T_I,
                    regulation_coeffs = [psi_N_I, psi_cM_I, psi_E_I, psi_E_c],
                    infection = "prim"):
    
    dt =  duration/steps
    
    # draw population of reponding cells for agent-based simulations
    mu_rates = np.array([g_NE_max + g_NM_max, b_E_max + g_EM_max, g_ME_max, b_c1])
    cv_rates = np.array([0.1, 0.1, 0.1, 0.5])

    div_rates = (np.random.lognormal(mean = np.log(mu_rates/np.sqrt(1 + cv_rates**2)), 
                        sigma = np.sqrt(np.log(1+ cv_rates**2)), size = (N0,4))).T
    
    # define variables for storage
    N_m = np.zeros((steps+1, N0))
    N_m[0,:] +=1

    S = np.zeros(steps+1)
    E_m = np.zeros((steps+1, N0))
    cM_m = np.zeros((steps+1, N0))
    eM_m = np.zeros((steps+1, N0))
    I = np.zeros(steps+1)

    # Run population simulation
    t = 0.0
    I[0] = I_0
    S[0] = S0
    
    for i in np.arange(1, steps + 1):
        # compute total population of cell types
        E_pop, cM_pop, eM_pop = np.sum(E_m[i-1]), np.sum(cM_m[i-1]), np.sum(eM_m[i-1])
        p_t = p_A(t1,I[i-1])
        hl_t = hl_u(c1_ss(t1,I[i-1], E_pop, Treg0, np.mean(div_rates[3])), k_E)
        
        # run infection dynamics: replication and effector clearance
        S[i] = S[i-1] - dt*(I[i-1] >= I_0)*np.exp(-t/T_I)*I[i-1]*b_I*(S[i-1]-I[i-1])
        I[i] = I[i-1] + dt*((I[i-1] >= I_0)*np.exp(-t/T_I)*I[i-1]*b_I*(S[i-1]-I[i-1]) - d_IE*I[i-1]*E_pop - d_I*I[i-1])

        # naive cells have a timer to activation
        N_act = (np.random.exponential(np.nan_to_num(1/g_N(t1,I[i-1]), E_pop, Treg0, max_val = div_rates[0,:]), N0) < dt)*N_m[i-1,:]
        N_to_cM = np.random.binomial(N_act.astype(int), np.nan_to_num((1-psi_N_I)*(1-p_t) + psi_N_I*p_t), N0)

        # central memory cells divide and differentiate
        cM_div = np.random.binomial(cM_m[i-1,:].astype(int), np.nan_to_num(dt*b_cM(t1,I[i-1], E_pop, Treg0, max_val = div_rates[2])), N0)
        cM_to_E = np.random.binomial(2*cM_div.astype(int), np.nan_to_num((1-psi_N_I)*(1-p_t) + psi_N_I*p_t), N0)

        # effector cells divide and differentiate, or die, or both
        E_div = np.random.binomial(-np.minimum(-E_m[i-1,:].astype(int),0), np.nan_to_num(dt*b_E(t1,I[i-1], E_pop, Treg0, max_val = div_rates[1], b_c1_pop = np.mean(div_rates[3]))), N0)
        E_to_eM = np.random.binomial(E_div.astype(int), np.nan_to_num(alpha*((1-psi_E_I)*(1-p_t) + psi_E_I*p_t) + (1-alpha)*((1 - psi_E_c)*(1-hl_t) + psi_E_c*hl_t)), N0)
        E_death = np.random.binomial(-np.minimum(-E_m[i-1,:].astype(int),0), np.nan_to_num(dt*d_E(t1,I[i-1], E_pop, Treg0, b_E_pop = div_rates[2], b_c1_pop = np.mean(div_rates[3]))), N0)

        E_to_M_death = np.random.binomial(E_to_eM.astype(int), np.nan_to_num(dt*d_E(t1,I[i-1], E_pop, Treg0, b_E_pop = div_rates[2], b_c1_pop = np.mean(div_rates[3]))), N0)


        # Update population dynamics
        N_m[i,:] = N_m[i-1,:] - N_act
        cM_m[i,:] = N_to_cM + (cM_m[i-1,:] - cM_div + 2*cM_div - cM_to_E) 
        E_m[i,:] = (N_act - N_to_cM) + cM_to_E + (E_m[i-1,:] + E_div - 2*E_to_eM) - (E_death - E_to_M_death) # sometimes get negative E value. Must be a bug I'm not seeing.
        eM_m[i,:] = eM_m[i-1,:] + (E_to_eM - E_to_M_death)

        # increment time
        t += dt
        
        # Collect population dynamics
        N, cM, E, eM = np.sum(N_m, axis = 1), np.sum(cM_m, axis = 1), np.sum(E_m, axis = 1), np.sum(eM_m, axis = 1)
        ts = np.linspace(0, duration, steps + 1)
    
    return np.array(regulation_coeffs), I, N, E, cM, eM, Treg0, ts

### (4) Parallelize simulation runs
def sum_sim(I_0 = I_0, b_I = b_I, t1 = t1, d_IE = d_IE, T_I = T_I,
            rates = [b_E_max, b_c1, g_NE_max, g_NM_max, g_ME_max, g_EM_max],
            noise_cv = [0.1,0.0,0.0,0.0,0.0,0.0],
            infection = "primary",
            sim_kind = "agent"):
    # compute state and costate dynamics
    dt = duration/steps
    
    if sim_kind == "agent":
        rates, I, N, E, cM, eM, T, ts = agent_stoch_sim(I_0, b_I, t1, d_IE, T_I,
                                        regulation_coeffs = rates,
                                        infection = infection)
    else:
        rates, I, N, E, cM, eM, T, ts = stoch_sim(I_0, b_I, t1, d_IE, T_I,
                                            rates = rates,
                                            noise_cv = noise_cv,
                                            infection = infection)
    
    run_data = np.concatenate((rates, [I_0, b_I, t1, d_IE, np.sum(np.log(np.maximum(I[0:np.argmax(E)], E_min))/np.argmax(E))*np.argmax(I)*dt, np.argmax(I)*dt,
                                       np.where(np.argmax(E) < 1, 0.0, np.max(E)/(np.argmax(E)*dt)), \
                                       np.argmax(E)*dt, np.where(np.max(E) < 1.0, 0.0, (eM[-1]+cM[-1])/np.max(E)), np.sum(E*dt), np.sum(np.log(np.maximum(E, E_min))*dt),\
                                       np.sum(np.log(np.maximum(E+eM+cM, E_min))*dt)]), 
                              axis = None)
    
    return run_data

stat_names = [r"$b_E$", r"$b_{c_1}$", r"$g_{NE}$", r"$g_{NM}$", r"$g_{ME}$", r"$g_{EM}$",\
              r"$I_0$", r"$b_{I}$", r"$t_1$", r"$d_{I,E}$",\
              r"$T_{I}^{max}\int_0^{T_{E}^{max}} \frac{\log(I)}{T_{E}^{max}} dt$",\
              r"$T_{I}^{max}$", r"$\frac{E^{max}}{T_{E}^{max}}$", r"$T_{E}^{max}$",\
              r"$\frac{(cM + eM)^\infty}{E^{max}}$", r"$\int E dt$", \
              r"$\int \log\left(E\right) dt$", r"$\int \log\left(E+M\right) dt$"]

stat_names_for_df = ['I_0','b_I', 't_1','d_IE',\
                     'int_logI','T_I_max', 'E_max_T_E_max',\
                     'T_E_max','mem_frac','int_logE', 'int_logEM']

### (5) define basic mutual information function
from sklearn.metrics import mutual_info_score

def calc_MI(x, y, bin_num = 100, correction = True):
    _, bx = pd.qcut(x, bin_num, retbins=True, duplicates = 'drop')
    _, by = pd.qcut(y, bin_num, retbins=True, duplicates = 'drop')
    
    if bx.size == 1:
        bx = np.append(bx, bx +1)
        
    if by.size == 1:
        by = np.append(by, by +1)
    
    c_xy = np.histogram2d(x, y, (bx,by))[0]
    mi_raw = mutual_info_score(None, None, contingency=c_xy)/np.log(2)
    
    # MI correction by shuffling data
    c_xy_shuffle = np.histogram2d(x, y[np.random.permutation(y.shape[0])], (bx,by))[0]
    mi_correction = mutual_info_score(None, None, contingency=c_xy_shuffle)/np.log(2)
    
    if correction == True:
        out = mi_raw - mi_correction
    else:
        out = mi_raw
    
    return out