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
b_I = 8
tau_I = 10
d_IE = 3.8*10**(-4) # proxy for virulence
d_I = 1*10**(-1)
T_I = 2
S0 = 10_000_000 #susceptible cells

t2 = 2 # dissociation time
n, m = 4, 2

# cells
N0 = 100
Treg0 = 0 # initial Tregs
b_N_max = 1.50
b_E_max = 3.0 + 0.07
b_cM_max = 1

E_min = 1 # minimum detectable cell counts

d_E_max = 2.0
b_eTr_max = 10000
d_eTr_max = 1

# cytokines
l = 2
b_c1 = 1000*3600*24
b_c2 = 10 #b_c1/100
f_T = 1.4*10**(7)*10**(4)*3600*24/(6.0221408*10**23) #*50*10**(-6))
k_c2_a = 10**(-14)*(6.0221408*10**23)*50*10**(-6)
tau_c = 0.015*24
I_c1 = 0
I_c2 = 0

# cellular cytokine thresholds
k_E = 10**(-11)*(6.0221408*10**23)*50*10**(-6)
k_M = k_E
k_Tr = k_E/2

# set initial state
init_state = np.array([N0, 0, 0, 0, Treg0]) # N, E, cM, eM, T

# activation threshold
mean_theta = 50_000_000 # set to point at which activation probabity equals antigen frequency
cv_theta = 0.5 #np.log((mu_theta*tau_I**n +t2**n)/(mu_theta*tau_I**m + t2**m))/10

# hyper parameters
alpha = 0.5 # antigen-cyokine weighting
psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c = 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 # decision to upregulate or downregulate based on stimulus

### (2) Define functions for simulations
# Define functions for simulations
@nb.vectorize([nb.float64(nb.float64)])
def verf(x):
    return erf(x)

@nb.njit
def hl_u(x,k):
    return (x**l)/(k**l + x**l)

@nb.njit
def a_1(tau_I,I):
    return 20*I

@nb.njit
def a_stim(tau_I,I):
    return (a_1(tau_I,I)/a2*tau_I**n +t2**n)/(a_1(tau_I,I)/a2*tau_I**m + t2**m)

@nb.njit
def p_A(tau_I, I, mu = np.log(a_stim(4*t2, mean_theta)/np.sqrt(1 + cv_theta**2)), sigma = np.sqrt(np.log(1 + cv_theta**2))):
    
    # mu = np.log(a_stim(4*t2, mean_theta)/np.sqrt(1 + cv_theta**2))
    # sigma = np.sqrt(np.log(1 + cv_theta**2))
    
    return 1/2 + verf((np.log(a_stim(tau_I,I))-mu)/np.sqrt(2*sigma**2))/2

@nb.njit
def c2_ss(I, E):
    
    return E*b_c2*tau_c

@nb.njit
def c1_ss(tau_I,I, E, eTr, b_c1_pop = b_c1):
    
    return (E*b_c1_pop*p_A(tau_I,I))/(eTr*f_T + 1/tau_c)

@nb.njit
def b_N(tau_I,I,E,T, max_val = b_N_max):
    
    return max_val*p_A(tau_I,I)

@nb.njit
def g_NE(tau_I,I,E,T, max_val = b_N_max, b_c1_pop = b_c1, psi_N_I = psi_N_I, psi_N_c = psi_N_c):
    
    return max_val*p_A(tau_I,I)

@nb.njit
def g_NM(tau_I,I,E,T, max_val = b_N_max, b_c1_pop = b_c1, psi_N_I = psi_N_I, psi_N_c = psi_N_c):
    
    out = max_val*(alpha*((1-psi_N_I)*(1-p_A(tau_I,I)) + psi_N_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_N_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_N_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
    
    return out

@nb.njit
def b_E(tau_I,I, E, T, max_val = b_E_max, b_c1_pop = b_c1):
    
    return max_val*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)

@nb.njit
def g_EE(tau_I,I, E, T, max_val = b_E_max, b_c1_pop = b_c1, psi_E_I = psi_E_I, psi_E_c = psi_E_c):
    
    out = max_val*(alpha*((1-psi_E_I)*(1-p_A(tau_I,I)) + psi_E_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_E_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_E_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
        
    return out

@nb.njit
def b_eTr(tau_I,I, E, eTr, max_val = b_eTr_max, b_c1_pop = b_c1):
    
    mu_eTr = np.log(a_stim(t2, mean_theta)/np.sqrt(1 + cv_theta**2))
    sigma_eTr = np.sqrt(np.log(1 + (10*cv_theta)**2))
    return max_val*(p_A(tau_I,I, mu = mu_eTr, sigma = sigma_eTr))*hl_u(c1_ss(tau_I,I, E, eTr, b_c1_pop), k_E)

@nb.njit
def b_cM(tau_I,I,E,T, max_val = b_cM_max):

    return max_val*p_A(tau_I,I)

@nb.njit
def g_MM(tau_I,I,E,T, max_val = b_cM_max, b_c1_pop = b_c1, psi_cM_I = psi_cM_I, psi_cM_c = psi_cM_c):
    
    out = max_val*(alpha*((1-psi_cM_I)*(1-p_A(tau_I,I)) + psi_cM_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_cM_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_cM_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
    
    return out

@nb.njit
def g_EM(tau_I,I,E,T, max_val = b_E_max):
    
    return max_val*(1-p_A(tau_I,I))

@nb.njit
def d_E(tau_I,I, E, T, b_E_pop = b_E_max, b_c1_pop = b_c1):
    
    return d_E_max*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E))

@nb.njit
def d_eTr(tau_I,I, E, T, max_val = d_eTr_max, b_c1_pop = b_c1):
    
    return d_eTr_max*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_Tr))

# define dynamics
@nb.njit
def pop_state_dyn(t, z, I_0, b_I, tau_I, d_IE, T_I, 
                  bN_pop, bE_pop, bcM_pop, bc1_pop,
                  psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c,
                  infection = "prim"):
        
    if infection == "prim":
        I, N, E, cM, eM, eTr = z

        bN = b_N(tau_I,I,E,eTr, bN_pop)
        gNM = g_NM(tau_I,I,E,eTr, max_val = bN, b_c1_pop = bc1_pop, psi_N_I = psi_N_I, psi_N_c = psi_N_c)
        bcM = b_cM(tau_I,I,E,eTr, max_val = bcM_pop)
        gMM = g_MM(tau_I,I,E,eTr, max_val = bcM, b_c1_pop = bc1_pop, psi_cM_I = psi_cM_I, psi_cM_c = psi_cM_c)
        bE = b_E(tau_I,I, E, eTr, max_val = bE_pop, b_c1_pop = bc1_pop)
        gEE = g_EE(tau_I,I, E, eTr, max_val = bE, b_c1_pop = bc1_pop, psi_E_I = psi_E_I, psi_E_c = psi_E_c)
        dE = d_E(tau_I, I, E, eTr, bE, bc1_pop)
        beTr = b_eTr(tau_I,I, E, eTr, max_val = b_eTr_max, b_c1_pop = bc1_pop)
        deTr = d_eTr(tau_I,I, E, eTr, max_val = d_eTr_max, b_c1_pop = bc1_pop)
        
        out = np.asarray([(I >= I_0/10)*np.exp(-t/T_I)*I*b_I - d_IE*I*E - d_I*I,\
                          -bN*N,\
                          (bN - gNM)*N + cM*(bcM - gMM) +  E*(2*gEE -bE - dE), \
                          gNM*N + cM*(2*gMM -bcM), \
                          E*(bE - gEE),\
                          beTr - eTr*deTr])
        
    elif infection == "sec":
        I, N, E, pM, cM, eM, eTr = z

        bN = b_N(tau_I,I,E,eTr, bN_pop)
        gNM = g_NM(tau_I,I,E,eTr, max_val = bN, b_c1_pop = bc1_pop, psi_N_I = psi_N_I, psi_N_c = psi_N_c)
        bcM = b_cM(tau_I,I,E,eTr, max_val = bcM_pop)
        gMM = g_MM(tau_I,I,E,eTr, max_val = bcM, b_c1_pop = bc1_pop, psi_cM_I = psi_cM_I, psi_cM_c = psi_cM_c)
        bE = b_E(tau_I,I, E, eTr, max_val = bE_pop, b_c1_pop = bc1_pop)
        gEE = g_EE(tau_I,I, E, eTr, max_val = bE, b_c1_pop = bc1_pop, psi_E_I = psi_E_I, psi_E_c = psi_E_c)
        dE = d_E(tau_I, I, E, eTr, bE, bc1_pop)
        beTr = b_eTr(tau_I,I, E, eTr, max_val = b_eTr_max, b_c1_pop = bc1_pop)
        deTr = d_eTr(tau_I,I, E, eTr, max_val = d_eTr_max, b_c1_pop = bc1_pop)
        
        out = np.asarray([(I >= I_0/10)*np.exp(-t/T_I)*I*b_I - d_IE*I*E - d_I*I,\
            -bN*N,\
            (bN - gNM)*N + (pM+cM)*(bcM - gMM) +  E*(2*gEE -bE - dE), \
            pM*(2*gMM -bcM), \
            gNM*N + cM*(2*gMM -bcM), \
            E*(bE - gEE), \
            beTr - eTr*deTr])
    
    return out

### (3) Code to run individual simulation
# Run single simulation and plot outputs
duration = 20
steps = 10**4

## ODE-based model
def stoch_sim(I_0 = I_0, b_I = b_I, tau_I = tau_I, d_IE = d_IE, T_I = T_I,
              regulation_coeffs = [psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c],
              rates = [b_N_max, b_E_max, b_cM_max, b_c1],
              noise_model = "pop", 
              rate_cv = [0.5, 0.5, 0.5, 2.0], 
              infection = "prim", duration = 20, steps = 10**4):

    
    dt = duration/steps
    ts = np.linspace(0, duration, steps + 1)
    
    psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c = regulation_coeffs
    
    if noise_model == "pop":
        bN_pop = np.mean(np.minimum((rates[0] > 0)*np.random.lognormal(mean = np.log(rates[0]/np.sqrt(1 + rate_cv[0]**2) + (rates[0] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[0]**2)), size = N0), 10))
            
        bE_pop = np.mean(np.minimum((rates[1] > 0)*np.random.lognormal(mean = np.log(rates[1]/np.sqrt(1 + rate_cv[1]**2) + (rates[1] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[1]**2)), size = N0), 10))
        
        bcM_pop = np.mean(np.minimum((rates[2] > 0)*np.random.lognormal(mean = np.log(rates[2]/np.sqrt(1 + rate_cv[2]**2) + (rates[2] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[2]**2)), size = N0), 10))
        
        bc1_pop = np.mean(np.random.lognormal(mean = np.log(rates[3]/np.sqrt(1 + rate_cv[3]**2) + (rates[3] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[3]**2)), size = N0))
        
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([I_0], init_state), axis = None), method="Radau",
                    dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I, bN_pop, bE_pop, bcM_pop, bc1_pop, psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c]).sol(ts).T
    else:
        states = solve_ivp(state_dyn, [0, duration], np.concatenate(([I_0],init_state), axis = None), method="Radau", dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I]).sol(ts).T

    I, N, E, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]
    
    if infection == "sec" and noise_model == "pop":
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([I_0], np.array([N[-1], 0,cM[-1] + eM[-1], 0, 0, Treg0])), axis = None), method="Radau",
                    dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I, bN_pop, bE_pop, bcM_pop, bc1_pop, psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c, infection]).sol(ts).T
        
        I, N, E, pM, cM, eM, T = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5], states[:,6]
    else:
        pM = 0
    
    return np.array([bN_pop, bE_pop, bcM_pop, bc1_pop]), I, N, E, cM, eM + pM, T, ts

## agent-based simulation with tau leaping
def agent_stoch_sim(I_0 = I_0, b_I = b_I, tau_I = tau_I, d_IE = d_IE, T_I = T_I,
                    regulation_coeffs = [psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c],
                    rates = [b_N_max, b_E_max, b_cM_max, b_c1],
                    rate_cv = np.array([0.5, 0.5, 0.5, 2.0]),
                    infection = "prim", duration = 20, steps = 10**4):
    
    dt =  duration/steps
    
    # draw population of reponding cells for agent-based simulations
    psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c = regulation_coeffs
    
    bN = np.minimum(np.random.lognormal(mean = np.log((rates[0])/np.sqrt(1 + rate_cv[0]**2)), 
                        sigma = np.sqrt(np.log(1+ rate_cv[0]**2)), size = N0), 10)
    bE = np.minimum(np.random.lognormal(mean = np.log((rates[1])/np.sqrt(1 + rate_cv[1]**2)), 
                        sigma = np.sqrt(np.log(1+ rate_cv[1]**2)), size = N0), 10)
    bcM = np.minimum(np.random.lognormal(mean = np.log((rates[2])/np.sqrt(1 + rate_cv[2]**2)), 
                        sigma = np.sqrt(np.log(1+ rate_cv[2]**2)), size = N0), 10)
    bc1 = np.random.lognormal(mean = np.log((rates[3])/np.sqrt(1 + rate_cv[3]**2)), 
                        sigma = np.sqrt(np.log(1+ rate_cv[3]**2)), size = N0)
    
    
    # define variables for storage
    N_m = np.zeros((steps+1, N0))
    N_m[0,:] +=1

    # S = np.zeros(steps+1)
    E_m = np.zeros((steps+1, N0))
    cM_m = np.zeros((steps+1, N0))
    eM_m = np.zeros((steps+1, N0))
    I = np.zeros(steps+1)

    # Run population simulation
    t = 0.0
    I[0] = I_0
    # S[0] = S0
    
    for i in np.arange(1, steps + 1):
        # compute total population of cell types
        E_pop, cM_pop, eM_pop = np.sum(E_m[i-1]), np.sum(cM_m[i-1]), np.sum(eM_m[i-1])
        p_t = p_A(tau_I,I[i-1])
        hl_t = hl_u(c1_ss(tau_I,I[i-1], E_pop, Treg0, np.mean(bc1)), k_E)
        
        # run infection dynamics: replication and effector clearance
        # S[i] = S[i-1] - dt*(I[i-1] >= I_0)*np.exp(-t/T_I)*I[i-1]*b_I*(S[i-1]-I[i-1])
        I[i] = I[i-1] + dt*((I[i-1] >= I_0)*np.exp(-t/T_I)*I[i-1]*b_I - d_IE*I[i-1]*E_pop - d_I*I[i-1])

        # naive cells have a timer to activation
        N_act = (np.random.poisson(np.nan_to_num(dt*b_N(tau_I,I[i-1], E_pop, Treg0, max_val = bN)), N0) > 0)*N_m[i-1,:]
        N_to_cM = np.random.binomial(N_act.astype(int), 
                                     np.nan_to_num(alpha*((1-psi_N_I)*(1-p_t) + psi_N_I*p_t) + (1-alpha)*((1 - psi_N_c)*(1-hl_t) + psi_N_c*hl_t)), N0)

        # central memory cells divide and differentiate
        cM_div = np.random.binomial(cM_m[i-1,:].astype(int), np.nan_to_num(dt*b_cM(tau_I,I[i-1], E_pop, Treg0, max_val = bcM)), N0)
        cM_to_E = np.random.binomial(2*cM_div.astype(int), 
                                     np.nan_to_num(alpha*((1-psi_cM_I)*(1-p_t) + psi_cM_I*p_t) + (1-alpha)*((1 - psi_cM_c)*(1-hl_t) + psi_cM_c*hl_t)), N0)

        # effector cells divide and differentiate, or die, or both
        E_div_die = np.random.binomial(E_m[i-1,:].astype(int), np.nan_to_num(dt*b_E(tau_I,I[i-1], E_pop, Treg0, max_val = bE, b_c1_pop = np.mean(bc1))
                                       + dt*d_E(tau_I,I[i-1], E_pop, Treg0, b_E_pop = bcM, b_c1_pop = np.mean(bc1))), N0)
        E_div = np.random.binomial(E_div_die.astype(int), np.nan_to_num(b_E(tau_I,I[i-1], E_pop, Treg0, max_val = bE, b_c1_pop = np.mean(bc1))/(b_E(tau_I,I[i-1], E_pop, Treg0, max_val = bE, b_c1_pop = np.mean(bc1)) + d_E(tau_I,I[i-1], E_pop, Treg0, b_E_pop = bE, b_c1_pop = np.mean(bc1)))), N0)
        E_to_eM = np.random.binomial(2*E_div.astype(int), 
                                     np.nan_to_num(alpha*((1-psi_E_I)*(1-p_t) + psi_E_I*p_t) + (1-alpha)*((1 - psi_E_c)*(1-hl_t) + psi_E_c*hl_t)), N0)
        E_die = E_div_die - E_div


        # Update population dynamics
        N_m[i,:] = N_m[i-1,:] - N_act
        cM_m[i,:] = N_to_cM + (cM_m[i-1,:] + cM_div - cM_to_E) 
        E_m[i,:] = (N_act - N_to_cM) + cM_to_E + (E_m[i-1,:] + E_div - E_to_eM) - E_die  # sometimes get negative E value. Must be a bug I'm not seeing.
        eM_m[i,:] = eM_m[i-1,:] + E_to_eM

        # increment time
        t += dt
        
        # Collect population dynamics
        N, cM, E, eM = np.sum(N_m, axis = 1), np.sum(cM_m, axis = 1), np.sum(E_m, axis = 1), np.sum(eM_m, axis = 1)
        ts = np.linspace(0, duration, steps + 1)
    
    return np.array(regulation_coeffs), I, N, E, cM, eM, Treg0, ts

### (4) Parallelize simulation runs
def sum_sim(I_0 = I_0, b_I = b_I, tau_I = tau_I, d_IE = d_IE, T_I = T_I,
            regulation_coeffs = [psi_N_I, psi_N_c, psi_cM_I, psi_cM_c, psi_E_I, psi_E_c],
            rates = [b_N_max, b_E_max, b_cM_max, b_c1],
            rate_cv = [0.5, 0.5, 0.5, 2.0],
            infection = "primary",
            sim_kind = "agent"):
    # compute state and costate dynamics
    if sim_kind == "agent":
        rates, I, N, E, cM, eM, T, ts = agent_stoch_sim(I_0, b_I, tau_I, d_IE, T_I,
                                        regulation_coeffs = regulation_coeffs,
                                        rates = rates,
                                        rate_cv =  rate_cv,
                                        infection = infection)
    else:
        rates, I, N, E, cM, eM, T, ts = stoch_sim(I_0, b_I, tau_I, d_IE, T_I,
                                                  regulation_coeffs = regulation_coeffs,
                                                  rates = rates,
                                                  rate_cv = rate_cv,
                                                  infection = infection)
        
    dt = ts[1]-ts[0]
    
    run_data = np.concatenate((regulation_coeffs, [I_0, b_I, tau_I, d_IE, np.sum( np.log(np.maximum(I, E_min)) )*dt, np.argmax(I)*dt,
                                       np.where(np.argmax(E) < 1, 0.0, np.max(E)/(np.argmax(E)*dt)), \
                                       np.argmax(E)*dt, np.where(np.max(E) < 1.0, 0.0, (eM[-1]+cM[-1])/np.max(E)), np.sum(E*dt), np.sum(np.log(np.maximum(E, E_min))*dt),\
                                       np.sum(np.log(np.maximum(E+eM+cM, E_min))*dt)]), 
                              axis = None)
    
    return rates, run_data

stat_names = [r"$\psi_{N}^{(I)}$", r"$\psi_{N}^{(c)}$", r"$\psi_{cM}^{(I)}$", r"$\psi_{cM}^{(c)}$", r"$\psi_{E}^{(I)}$", r"$\psi_{E}^{(c)}$",\
              r"$I_0$", r"$b_{I}$", r"$t_1$", r"$d_{I,E}$",\
              r"$\int_0^{T_{sim}} \log(I) dt$",\
              r"$T_{I}^{max}$", r"$\frac{E^{max}}{T_{E}^{max}}$", r"$T_{E}^{max}$",\
              r"$\frac{(cM + eM)^\infty}{E^{max}}$", r"$\int E dt$", \
              r"$\int \log\left(E\right) dt$", r"$\int \log\left(E+M\right) dt$"]

stat_names_for_df = ['I_0','b_I', 't_1','d_IE',\
                     'int_logI','T_I_max', 'E_max_T_E_max',\
                     'T_E_max','mem_frac','int_logE', 'int_logEM']

### (5) define basic mutual information function
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression

def calc_MI(x, y, bin_num = 50, correction = False):
    
    subsample_size = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])*x.size
    replicates = 20
    
    mi_data = np.zeros((subsample_size.size*replicates, 2))
    entry = 0
    
    for sub in subsample_size:
        for i in np.arange(0,replicates):
            choice =  np.random.choice(int(sub), int(sub))
            x_, y_ = x[choice], y[choice]


            _, bx = pd.qcut(x_, bin_num, retbins=True, duplicates = 'drop')
            _, by = pd.qcut(y_, bin_num, retbins=True, duplicates = 'drop')

            if bx.size == 1:
                bx = np.append(bx, bx +1)

            if by.size == 1:
                by = np.append(by, by +1)

            c_xy = np.histogram2d(x_, y_, (bx,by))[0]
            mi_raw = mutual_info_score(None, None, contingency=c_xy)/np.log(2)

            # MI correction by shuffling data
            c_xy_shuffle = np.histogram2d(x_, y_[np.random.permutation(y_.shape[0])], (bx,by))[0]
            mi_correction = mutual_info_score(None, None, contingency=c_xy_shuffle)/np.log(2)

            if correction == True:
                out = mi_raw - mi_correction
            else:
                out = mi_raw
            
            mi_data[entry, 1], mi_data[entry, 0] = out, 1/sub
            
            entry += 1
            
    lr = LinearRegression()
    lr.fit(mi_data[:,0].reshape(-1,1), mi_data[:,1].reshape(-1,1))
    
    return lr.intercept_


# def calc_MI(x, y, bin_num = 10, correction = True):
    
#     subsample_size = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.1])*x.size
    
#     for sub in subsample_size:
#         x_, y_ = np.random.choice(x, int(sub)), np.random.choice(y, int(sub))
        
        
#     _, bx = pd.qcut(x, bin_num, retbins=True, duplicates = 'drop')
#     _, by = pd.qcut(y, bin_num, retbins=True, duplicates = 'drop')
    
#     if bx.size == 1:
#         bx = np.append(bx, bx +1)
        
#     if by.size == 1:
#         by = np.append(by, by +1)
    
#     c_xy = np.histogram2d(x, y, (bx,by))[0]
#     mi_raw = mutual_info_score(None, None, contingency=c_xy)/np.log(2)
    
#     # MI correction by shuffling data
#     c_xy_shuffle = np.histogram2d(x, y[np.random.permutation(y.shape[0])], (bx,by))[0]
#     mi_correction = mutual_info_score(None, None, contingency=c_xy_shuffle)/np.log(2)
    
#     if correction == True:
#         out = mi_raw - mi_correction
#     else:
#         out = mi_raw
    
#     return out