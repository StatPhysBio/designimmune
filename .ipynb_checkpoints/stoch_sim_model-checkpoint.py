import itertools

### Stochastic model dynamics ###
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import qmc
from scipy import special
from math import erf
import numba as nb
import os as os
import pandas as pd

### (1) Define simulation parameters
# Define simulation parameters

# infection dynamics
S_0 = 10_000_000 #susceptible cells
b_S = 1_00_000
d_S = 0.1
I_0 = 10 # initial detectable levelof infected cells
b_I = 1*(10**(-6))
d_IE = 12 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE = 7.8*10**3 # effector avidity (half-max) for infected cells at low infection concetrations (Chao et al. 2004)
d_I = 10**(-1)

# APC dynamics
Aout_0 = 6*(10**4)
d_A = 0.8

# Inflammatory response
H_0, H_max = 0.1, 1.0
K_IH = K_IE
d_H = 0.5
b_H = 6.0
d_IH = 1

# cells
N_0 = 100
Treg0 = 0 # initial Tregs
b_N_max = 0.62
b_N_act_max = 2.8
b_E_max = 2.8
b_cM_max = 1.2
b_eM_max = 1/60

# division timer
b_myc = 4.0*(10**3)
d_myc = np.log(2)*24/7
myc_thresh = 10**(2.6)

E_min = 1 # minimum detectable cell counts

d_E_max = 2.0
b_eTr_max = 5000
d_eTr_max = 1

# cytokines
l = 2
b_c1 = 10000*3600*24
b_c2 = 10 #b_c1/100
f_T = 6*1.4*10**(7)*10**(4)*3600*24/(6.0221408*10**23) #*50*10**(-6))
k_c2_a = 10**(-14)*(6.0221408*10**23)*50*10**(-6)
tau_c = 0.25
I_c1 = 0
I_c2 = 0

# cellular cytokine thresholds
k_E = 10**(-11)*(6.0221408*10**23)*50*10**(-6)
k_M = k_E
k_Tr = k_E/100

# set initial state
init_state = np.array([N_0, 0, 0, 0, Treg0]) # N, E, cM, eM, T

# hyper parameters
alpha = 0.5 # antigen-cyokine weighting
psis = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0]) # decision to upregulate or downregulate based on stimulus: psi_NE_I, psi_NE_c, psi_EeM_I, psi_EeM_c, psi_pME_I, psi_pME_c

### (2) Define functions for simulations
# Define functions for simulations
@nb.vectorize([nb.float64(nb.float64)])
def verf(x):
    return erf(x)

@nb.njit
def hl_u(x,k,l=l):
    return (x**l)/(k**l + x**l)

@nb.njit
def a_1(tau_I,I):
    return 20*I

@nb.njit
def a_stim(tau_I,I):
    return (a_1(tau_I,I)/a2*tau_I**n +t2**n)/(a_1(tau_I,I)/a2*tau_I**m + t2**m)

@nb.njit
def c2_ss(I, E):
    
    return E*b_c2*tau_c

@nb.njit
def c1_ss(p_act, tau_I,I, E, eTr, b_c1_pop = b_c1):
    #out = (E*b_c1_pop*p_A(tau_I,I))/(eTr*f_T + 1/tau_c)
    out = (E*b_c1_pop*p_act - k_Tr/tau_c + np.sqrt((E*b_c1_pop*p_act -k_Tr/tau_c)**2 + 4*(eTr*f_T + 1/tau_c)*k_Tr*E*b_c1_pop*p_act))/(2*(eTr*f_T + 1/tau_c))
    return out

@nb.njit
def b_N(tau_I,I,E,T, max_val = b_N_max):
    
    return max_val*p_A(tau_I,I)

@nb.njit
def b_N_act(tau_I,I,E,T, max_val = b_N_act_max):
    
    return max_val

@nb.njit
def g_NE(tau_I,I,E,T, max_val = b_N_max, b_c1_pop = b_c1, psi_NE_I = psis[0], psi_NE_c = psis[1]):
    
    return max_val*p_A(tau_I,I)

@nb.njit
def g_NM(tau_I,I,E,T, max_val = b_N_max, b_c1_pop = b_c1, psi_NE_I = psis[0], psi_NE_c = psis[1]):
    
    out = max_val*(alpha*((1-psi_NE_I)*(1-p_A(tau_I,I)) + psi_NE_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_NE_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_NE_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
    
    return out

@nb.njit
def b_E(tau_I,I, E, T, max_val = b_E_max, b_c1_pop = b_c1):
    
    return max_val*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E*(1-p_A(tau_I,I)))

@nb.njit
def g_EE(tau_I,I, E, T, max_val = b_E_max, b_c1_pop = b_c1, psi_EeM_I = psis[4], psi_EeM_c = psis[5]):
    
    out = max_val*(alpha*((1-psi_EeM_I)*(1-p_A(tau_I,I)) + psi_EeM_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_EeM_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_EeM_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
        
    return out

@nb.njit
def b_eTr(c1):
    
    return b_eTr_max*hl_u(c1, k_E)

@nb.njit
def b_cM(tau_I,I,E,T, max_val = b_cM_max, b_c1_pop = b_c1):

    return max_val*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)

@nb.njit
def g_MM(tau_I,I,E,T, max_val = b_cM_max, b_c1_pop = b_c1, psi_cME_I = psis[2], psi_cME_c = psis[3]):
    
    out = max_val*(alpha*((1-psi_cME_I)*(1-p_A(tau_I,I)) + psi_cME_I*p_A(tau_I,I)) + (1-alpha)*((1 - psi_cME_c)*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + psi_cME_c*hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)))
    
    return out

@nb.njit
def g_EM(tau_I,I,E,T, max_val = b_E_max):
    
    return max_val*(1-p_A(tau_I,I))

@nb.njit
def d_E(tau_I,I, E, T, b_E_pop = b_E_max, b_c1_pop = b_c1):
    
    return d_E_max*(1-hl_u(c1_ss(tau_I,I, E, T, b_c1_pop), k_E)) + b_E(tau_I,I, E, T, max_val = b_E_max, b_c1_pop = b_c1_pop)/20

@nb.njit
def d_eTr(c1):
    
    return d_eTr_max*(1-hl_u(c1, k_Tr))

#@nb.njit
def p_XtoY(I, H, psi_I, psi_H, F_0, K_I, K_H, reg_model = "mwc_like", alpha = 0.5):
    ### variable
    # I := antigenic stimuli
    # H := inflammatory stimuli
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towrads transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    # reg_model := family of regulatory functions considered: Monod-Wyman-Changeaux inspired, and Hill functions
    # alpha := relative weight of antigen and cytokine signals in Hill-OR model
    
    F_1 = psi_I*np.log(1 + I/K_I) + psi_H*np.log(1 + H/K_H)
    
    if reg_model == "mwc_like":
        out = 1/(1 + np.exp(- F_1 - F_0))
        
    elif reg_model == "hill_and":
        I_sig = I**(psi_I)/(K_I**(psi_I) + I**(psi_I))
        H_sig = H**(psi_H)/(K_H**(psi_H) + H**(psi_H))
        
        out = np.nan_to_num(I_sig, nan = 1.0)*np.nan_to_num(H_sig, nan = 1.0)
        
    elif reg_model == "hill_or":
        I_sig = I**(psi_I)/(K_I**(psi_I) + I**(psi_I))
        H_sig = H**(psi_H)/(K_H**(psi_H) + H**(psi_H))
        
        out = alpha*np.nan_to_num(I_sig, nan = 1.0) + (1-alpha)*np.nan_to_num(H_sig, nan = 1.0)
        
    return out


def init_list(length, size):
    
    out = [np.zeros(length, dtype = int) for l in np.arange(size)]
    return out

# define dynamics
@nb.njit
def pop_state_dyn(t, z, I_0, b_I, tau_I, d_IE, T_I, 
                  bN_pop, bE_pop, bcM_pop, bc1_pop,
                  psi_NE_I, psi_NE_c, psi_cME_I, psi_cME_c, psi_EeM_I, psi_EeM_c,
                  infection = "prim"):
        
    if infection == "prim":
        I, N, E, cM, eM, eTr = z

        bN = b_N(tau_I,I,E,eTr, bN_pop)
        gNM = g_NM(tau_I,I,E,eTr, max_val = bN, b_c1_pop = bc1_pop, psi_NE_I = psi_NE_I, psi_NE_c = psi_NE_c)
        bcM = b_cM(tau_I,I,E,eTr, max_val = bcM_pop)
        gMM = g_MM(tau_I,I,E,eTr, max_val = bcM, b_c1_pop = bc1_pop, psi_cME_I = psi_cME_I, psi_cME_c = psi_cME_c)
        bE = b_E(tau_I,I, E, eTr, max_val = bE_pop, b_c1_pop = bc1_pop)
        gEE = g_EE(tau_I,I, E, eTr, max_val = bE, b_c1_pop = bc1_pop, psi_EeM_I = psi_EeM_I, psi_EeM_c = psi_EeM_c)
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
        gNM = g_NM(tau_I,I,E,eTr, max_val = bN, b_c1_pop = bc1_pop, psi_NE_I = psi_NE_I, psi_NE_c = psi_NE_c)
        bcM = b_cM(tau_I,I,E,eTr, max_val = bcM_pop)
        gMM = g_MM(tau_I,I,E,eTr, max_val = bcM, b_c1_pop = bc1_pop, psi_cME_I = psi_cME_I, psi_cME_c = psi_cME_c)
        bE = b_E(tau_I,I, E, eTr, max_val = bE_pop, b_c1_pop = bc1_pop)
        gEE = g_EE(tau_I,I, E, eTr, max_val = bE, b_c1_pop = bc1_pop, psi_EeM_I = psi_EeM_I, psi_EeM_c = psi_EeM_c)
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
duration = 11
steps = 10**4

## ODE-based model
def stoch_sim(I_0 = I_0, b_I = b_I, N_0 = N_0, d_IE = d_IE, d_IH = d_IH,
              regulation_coeffs = psis,
              rates = [b_N_max, b_E_max, b_cM_max, b_c1],
              noise_model = "pop", 
              rate_cv = [0.5, 0.5, 0.5, 2.0], 
              infection = "prim", duration = 20, steps = 10**4):

    
    dt = duration/steps
    ts = np.linspace(0, duration, steps + 1)
    
    psi_NE_I, psi_NE_c, psi_cME_I, psi_cME_c, psi_EeM_I, psi_EeM_c = regulation_coeffs
    
    if noise_model == "pop":
        bN_pop = np.mean(np.minimum((rates[0] > 0)*np.random.lognormal(mean = np.log(rates[0]/np.sqrt(1 + rate_cv[0]**2) + (rates[0] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[0]**2)), size = N_0), 1))
            
        bE_pop = np.mean(np.minimum((rates[1] > 0)*np.random.lognormal(mean = np.log(rates[1]/np.sqrt(1 + rate_cv[1]**2) + (rates[1] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[1]**2)), size = N_0), 5))
        
        bcM_pop = np.mean(np.minimum((rates[2] > 0)*np.random.lognormal(mean = np.log(rates[2]/np.sqrt(1 + rate_cv[2]**2) + (rates[2] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[2]**2)), size = N_0), 2.5))
        
        bc1_pop = np.mean(np.random.lognormal(mean = np.log(rates[3]/np.sqrt(1 + rate_cv[3]**2) + (rates[3] == 0)), sigma = np.sqrt(np.log(1+ rate_cv[3]**2)), size = N_0))
        
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([I_0], init_state), axis = None), method="Radau",
                    dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I, bN_pop, bE_pop, bcM_pop, bc1_pop, psi_NE_I, psi_NE_c, psi_cME_I, psi_cME_c, psi_EeM_I, psi_EeM_c]).sol(ts).T
    else:
        states = solve_ivp(state_dyn, [0, duration], np.concatenate(([I_0],init_state), axis = None), method="Radau", dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I]).sol(ts).T

    I, N, E, cM, eM, eTr = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5]
    
    if infection == "sec" and noise_model == "pop":
        states = solve_ivp(pop_state_dyn, [0, duration], np.concatenate(([I_0], np.array([N[-1], 0,cM[-1] + eM[-1], 0, 0, Treg0])), axis = None), method="Radau",
                    dense_output=True, args=[I_0, b_I, tau_I, d_IE, T_I, bN_pop, bE_pop, bcM_pop, bc1_pop, psi_NE_I, psi_NE_c, psi_cME_I, psi_cME_c, psi_EeM_I, psi_EeM_c, infection]).sol(ts).T
        
        I, N, E, pM, cM, eM, eTr = states[:,0], states[:,1], states[:,2], states[:,3], states[:,4], states[:,5], states[:,6]
    else:
        pM = 0
    
    return np.array([bN_pop, bE_pop, bcM_pop, bc1_pop]), I, N, E, cM, eM + pM, eTr, ts

#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################
sim_duration = 12
sim_steps = 1*(10**4)

t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_eM_diff, t_E_out, t_E_die, t_E_cyt = 1/6, 3/4, 1/4, 1/3, 1/2, 2.0, 1.0, 3.5, 1.0

def agent_stoch_sim(I_0 = I_0, b_I = b_I, N_0 = N_0, d_IE = d_IE, d_IH = d_IH,
                    regulation_coeffs = psis,
                    char_times = [t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_eM_diff, t_E_out, t_E_die, t_E_cyt],
                    trans_steps = np.array([1.0, 3.0, 4.0, 4.0, 4.0, 3.0, 2.0, 4.0, 4.0]),
                    infection = "prim",
                    reg_model = "mwc_like",
                    duration = sim_duration, 
                    steps = sim_steps):
    
    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    
    # set infection scenario: primary or secondary
    infection_count = 0
    if infection == "prim":
        infection_count = 1
    elif infection == "sec":
        infection_count = 2
    
    # draw population of reponding cells for agent-based simulations
    psi_NE_I, psi_NE_c, psi_EeM_I, psi_EeM_c, psi_pME_I, psi_pME_c = regulation_coeffs
    
    p_tcr = np.random.uniform(low = 0.5, high = 1.0, size = N_0_var)
    p_cyt = np.random.uniform(low = 0.8, high = 1.0, size = N_0_var)
    
    max_Na = 4 # maximum population of activated naive cells before fate specification
    
    for k in np.arange(0, infection_count):
        
        # define variables for storage
        I = np.zeros(steps+1)
        S = np.zeros(steps+1)
        eTr = np.zeros(steps+1)
        Aout = np.zeros(steps+1)
        Ain = np.zeros(steps+1)
        H = np.zeros(steps+1)
        
        if k == 0: # primary infection
            N_m = np.zeros((steps+1, N_0_var), dtype = int)
            N_m[0,:] +=1
            pM_m = np.zeros((steps+1, N_0_var), dtype = int)
        elif k == 1: # secondary infection
            N_m[0,:] = N_m[-1,:]
            pM_m[0,:] = cM_m[-1,:] + eM_m[-1,:]
        
        Na_m = np.zeros((steps+1, N_0_var), dtype = int)
        pMa_m = np.zeros((steps+1, N_0_var), dtype = int)
        
        mycNa_m = np.zeros((steps+1, N_0_var))
        mycEin_m = np.zeros((steps+1, N_0_var))
        mycEout_m = np.zeros((steps+1, N_0_var))
        myccM_m = np.zeros((steps+1, N_0_var))
        
        Ein_m = np.zeros((steps+1, N_0_var), dtype = int) # effector in lympoid organ
        Eout_m = np.zeros((steps+1, N_0_var), dtype = int) # effector in periphary
        cM_m = np.zeros((steps+1, N_0_var), dtype = int)
        eM_m = np.zeros((steps+1, N_0_var), dtype = int)
        p_XE = np.zeros((steps+1, 4))
        
        # Define event timer variables
        unbind_Na_timer = np.zeros(N_0_var)
        unbound_Na = np.zeros(N_0_var)
        
        div_Na_timer = init_list(0, N_0_var)
        
        diff_Na_E_timer = init_list(0, N_0_var)
        
        div_cM_timer = init_list(0, N_0_var)
        
        div_pMa_timer = [np.zeros(pM_m[0,l], dtype = int) for l in np.arange(N_0_var)]
        
        diff_pMa_timer = [np.zeros(pM_m[0,l], dtype = int) for l in np.arange(N_0_var)]
        
        div_Ein_timer = init_list(0, N_0_var)
        
        div_Eout_timer = init_list(0, N_0_var)
        
        cyt_Ein_timer = init_list(0, N_0_var)
        
        cyt_Eout_timer = init_list(0, N_0_var)
        
        out_Ein_timer = init_list(0, N_0_var)
        
        diff_Ein_eM_timer = init_list(0, N_0_var)
        
        diff_Eout_eM_timer = init_list(0, N_0_var)
        
        die_Ein_timer = init_list(0, N_0_var)
        
        die_Eout_timer = init_list(0, N_0_var)
        
        bound_IEin = init_list(0, N_0_var)
        bound_IEout = init_list(0, N_0_var)
        bound_IcM = init_list(0, N_0_var)
        bound_IpMa = [np.zeros(pM_m[0,l], dtype = int) for l in np.arange(N_0_var)]
        
        mycNa = init_list(0, N_0_var)
        mycEin = init_list(0, N_0_var)
        mycEout = init_list(0, N_0_var)
        myccM = init_list(0, N_0_var)
        
        p_NaE = np.zeros(N_0_var)
        p_EineM = np.zeros(N_0_var)
        p_EouteM = np.zeros(N_0_var)
        p_pME = np.zeros(N_0_var)
        
        b_stim_t = np.zeros(N_0_var)
        b_IEin_bind = np.zeros(N_0_var)
        b_IEout_bind = np.zeros(N_0_var)
        b_IN_bind = np.zeros(N_0_var)
        b_IcM_bind = np.zeros(N_0_var)
        b_IpMa_bind = np.zeros(N_0_var)
        b_act_t = np.zeros(N_0_var)
        b_Na_div = np.zeros(N_0_var)
        b_NaE_diff = np.zeros(N_0_var)
        b_E_div = np.zeros(N_0_var)
        b_cM_div = np.zeros(N_0_var)
        b_EineM_diff = np.zeros(N_0_var)
        b_EouteM_diff = np.zeros(N_0_var)
        b_pMa_diff = np.zeros(N_0_var)
        b_E_out = np.zeros(N_0_var)
        d_E_die = np.zeros(N_0_var)
        b_E_cyt = np.zeros(N_0_var)
        
        if k == 1: # cytolytic function is achieved almost instantly during secondary infection by memory
            cyt_E = np.ones(N_0_var)*(pM_m[0,:] > 0)
        
        #################################
        ### RUN POPULATION SIMULATION ###
        #################################
        t = 0.0
        S[0] = S_0
        I[0] = I_0
        Aout[0] = Aout_0
        H[0] = H_0
        
        # errors and troubleshooting
        error_time = 0

        for i in np.arange(1, steps + 1):
            # Compute total population of cell types
            Na_pop, Ein_pop, Eout_pop, cM_pop, eM_pop, pMa_pop = np.sum(Na_m[i-1]), np.sum(Ein_m[i-1]), np.sum(Eout_m[i-1]), np.sum(cM_m[i-1]), np.sum(eM_m[i-1]), np.sum(pMa_m[i-1])
            
            ## I. Recruitment/Priming

            # Phase 1: Naive cells encounter and bind APCs
            act_N = [np.random.binomial(N_m[i-1, j], b_act_t[j]*dt, 1) for j in np.arange(N_0_var)]

            # Phase 2: Activated naive cells are bound to APCs and receive stimulation
            # See section with binding times
            
            # Phase 3: Unbound activated naive cells divide
            Na_div_flag = [1*(Na_m[i-1,j] < max_Na) if Na_m[i-1,j] > 0 else 1 for j in np.arange(N_0_var)]

            div_Na_timer = [div_Na_timer[j] + np.random.binomial(unbound_Na[j], dt*b_Na_div[j]*trans_steps[2], Na_m[i-1,j]) if Na_m[i-1,j]*Na_div_flag[j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            div_Na = [(div_Na_timer[j] >= trans_steps[2])*Na_div_flag[j] if Na_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
                
            # After dividing, activated naive cells can differentiate
            diff_Na_E_timer = [diff_Na_E_timer[j] + np.random.binomial(1, dt*b_NaE_diff[j]*trans_steps[1], Na_m[i-1,j]) if Na_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            diff_Na_E = [1*(diff_Na_E_timer[j] >= int(2*trans_steps[1]/3)) if Na_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            ## II. Expansion

            # (a) New central memory cells divide
            div_cM_timer = [div_cM_timer[j] + np.random.binomial(1, dt*b_cM_div[j]*trans_steps[4], cM_m[i-1,j])*(myccM[j] > myc_thresh) if cM_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            div_cM = [1*(div_cM_timer[j] >= trans_steps[4]) if cM_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            # (b) Memory cells from a prior infection activate quickly and divide
            act_pM = [np.random.binomial(1*(pM_m[i-1,j] > 0), b_act_t[j]*dt, np.maximum(1,pM_m[i-1,j])) if pM_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            div_pMa_timer = [div_pMa_timer[j] + np.random.binomial(1*(pM_m[i-1,j] > 0), dt*b_Na_div[j]*trans_steps[2], pM_m[i-1,j]) if pMa_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            div_pMa_E = [1*(div_pMa_timer[j] >= trans_steps[2]) if pMa_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            diff_pMa_timer = [diff_pMa_timer[j] + np.random.binomial(1*(pM_m[i-1,j] > 0), dt*b_pMa_diff[j]*trans_steps[2], pM_m[i-1,j]) if pMa_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            diff_pMa_E = [1*(diff_pMa_timer[j] >= trans_steps[2]) if pMa_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            # (c) Effector cells divide, differentiate, gain cytolytic function, die
            div_Ein_timer = [div_Ein_timer[j] + np.random.binomial(1, dt*b_E_div[j]*trans_steps[3], Ein_m[i-1,j])*(mycEin[j] > myc_thresh) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            div_Ein = [1*(div_Ein_timer[j] >= trans_steps[3]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            cyt_Ein_timer = [cyt_Ein_timer[j] + np.random.binomial(1, dt*b_E_cyt[j]*trans_steps[8], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            cyt_Ein = [1*(cyt_Ein_timer[j] >= trans_steps[8]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            out_Ein_timer = [out_Ein_timer[j] + np.random.binomial(1, dt*b_E_out[j]*trans_steps[6], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            out_Ein = [(out_Ein_timer[j] >= trans_steps[6]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            diff_Ein_eM_timer = [diff_Ein_eM_timer[j] + np.random.binomial(1, dt*b_EineM_diff[j]*trans_steps[5], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            diff_Ein_eM = [1*(diff_Ein_eM_timer[j] >= trans_steps[5]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            die_Ein_timer = [die_Ein_timer[j] + np.random.binomial(1, dt*d_E_die[j]*trans_steps[7], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            die_Ein = [1*(die_Ein_timer[j] >= trans_steps[7]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            cyt_Eout_timer = [cyt_Eout_timer[j] + np.random.binomial(1, dt*b_E_cyt[j]*trans_steps[8], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            cyt_Eout = [1*(cyt_Eout_timer[j] >= trans_steps[8]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            diff_Eout_eM_timer = [diff_Eout_eM_timer[j] + np.random.binomial(1, dt*b_EouteM_diff[j]*trans_steps[5], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            diff_Eout_eM = [1*(diff_Eout_eM_timer[j] >= trans_steps[5]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            div_Eout_timer = [div_Eout_timer[j] + np.random.binomial(1, dt*b_E_div[j]*trans_steps[3], Eout_m[i-1,j])* (mycEout[j] > myc_thresh) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            div_Eout = [1*(div_Eout_timer[j] >= trans_steps[3]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]

            die_Eout_timer = [die_Eout_timer[j] + np.random.binomial(1, dt*d_E_die[j]*trans_steps[7], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            die_Eout = [1*(die_Eout_timer[j] >= trans_steps[7]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int) for j in np.arange(N_0_var)]
            
            #### Run infection dynamics: replication and effector clearance ####
            # Define population dependent CTL killing rate
            d_IEout_pop = 0.0
            d_IEin_pop = 0.0
            if Eout_pop > 0:
                d_IEout_pop = d_IE*np.mean(np.hstack([cyt_Eout[j]*p_tcr[j] for j in np.arange(N_0_var)]))
               
            if Ein_pop > 0:
                d_IEin_pop = d_IE*np.mean(np.hstack([cyt_Ein[j]*p_tcr[j] for j in np.arange(N_0_var)]))
                
            # check simulation stopping conditions
            if I[i-1] <= E_min and (Ein_pop + Eout_pop) <= E_min:
                break
            
            # Update state of susceptible, infected, APCs, and inflammation
            S[i] = S[i-1] + dt*(b_S - d_S*S[i-1] - b_I*S[i-1]*I[i-1])*(S[i-1] >= 0.0)
            I[i] = I[i-1] + dt*((I[i-1] >= I_0)*b_I*S[i-1]*I[i-1] - d_IH*I[i-1]*H[i-1] - d_IEout_pop*I[i-1]*(Eout_pop + eM_pop)/(K_IE + I[i-1] + Eout_pop + eM_pop) - d_I*I[i-1])*(I[i-1] >= 0.0)
            Aout[i] = Aout[i-1] - b_I*I[i-1]*Aout[i-1]*dt*(Aout[i-1] >= 0.0)
            Ain[i] = Ain[i-1] + dt*(b_I*I[i-1]*Aout[i-1] - d_A*Ain[i-1] - d_IEin_pop*Ain[i-1]*(Ein_pop + cM_pop)/(K_IE + Ain[i-1] + Ein_pop + cM_pop))*(Ain[i-1] >= 0.0)
            H[i] = H[i-1] + dt*(b_H*I[i-1]*(H_max-H[i-1])/(K_IH + I[i-1]) - d_H*(H[i-1]-H_0))*(H[i-1] >= 0.0)
            
            # Set negative values to zero, in 
            if (Ain[i-1] < 0.0 or Ain[i] < 0.0):
                # if error_time == 0:
                #     print("1. Error: Negative APCs {}".format(Ain[i-1]))
                Ain[i-1], Ain[i] = 0.0, 0.0
                error_time = i*dt
                if (Aout[i-1] < 0.0 or Aout[i] < 0.0):
                    Aout[i-1], Aout[i] = 0.0, 0.0
            
            #### Update population dynamics: implicit is that differentiation supercedes death if they coincide ####
            N_m[i] = N_m[i-1] - np.array([np.sum(act_N[j]) for j in np.arange(N_0_var)])
            Na_m[i] = Na_m[i-1] + np.array([np.sum(div_Na[j]) + np.sum(act_N[j]) - (1 - Na_div_flag[j])*max_Na for j in np.arange(N_0_var)])
            pM_m[i] = pM_m[i-1] - np.array([np.sum(act_pM[j]) for j in np.arange(N_0_var)])
            pMa_m[i] = pMa_m[i-1] - np.array([np.sum(div_pMa_E[j]) + np.sum(act_pM[j]) for j in np.arange(N_0_var)])
            cM_m[i] = cM_m[i-1] + np.array([np.sum(div_cM[j]) +  np.sum((1 - diff_Na_E[j])*(1 - Na_div_flag[j])) + np.sum(2*div_pMa_E[j] - diff_pMa_E[j]) for j in np.arange(N_0_var)])
            Ein_m[i] = Ein_m[i-1] + np.array([np.sum(div_Ein[j]) + np.sum(diff_Na_E[j])*(1 - Na_div_flag[j]) + np.sum(diff_pMa_E[j]) - np.sum(diff_Ein_eM[j] + out_Ein[j] + die_Ein[j] > 0) for j in np.arange(N_0_var)])
            Eout_m[i] = Eout_m[i-1] + np.array([np.sum(div_Eout[j]) + np.sum(out_Ein[j]*(1 - die_Ein[j])) - np.sum(die_Eout[j] + diff_Eout_eM[j] > 0) for j in np.arange(N_0_var)])
            eM_m[i] = eM_m[i-1] + np.array([np.sum(diff_Eout_eM[j]) + np.sum(diff_Ein_eM[j]) for j in np.arange(N_0_var)])
            
            #### Update and refresh timer variables for division ####
            div_Na_timer = [np.hstack( [div_Na_timer[j], np.zeros(act_N[j], dtype = int), div_Na_timer[j][div_Na[j] == 1]] ) % trans_steps[2] for j in np.arange(N_0_var)]
            
            diff_Na_E_timer = [np.hstack( [diff_Na_E_timer[j], np.zeros(act_N[j], dtype = int), diff_Na_E_timer[j][div_Na[j] == 1]] ) for j in np.arange(N_0_var)]
            
            myccM = [np.hstack( [myccM[j], myccM[j][div_cM[j]  > 0], mycNa[j][(1 - diff_Na_E[j])*(1 - Na_div_flag[j]) > 0]] ) for j in np.arange(N_0_var)]
            
            div_cM_timer = [np.hstack( [div_cM_timer[j], div_cM_timer[j][div_cM[j]  > 0], np.zeros(np.sum(2*div_pMa_E[j] - diff_pMa_E[j], dtype = int) + np.sum((1 - diff_Na_E[j])*(1 - Na_div_flag[j]), dtype = int))] ) % trans_steps[4] for j in np.arange(N_0_var)]

            div_pMa_timer = [div_pMa_timer[j][div_pMa_E[j] == 0] % trans_steps[2] for j in np.arange(N_0_var)]

            diff_pMa_timer = [diff_pMa_timer[j][div_pMa_E[j] == 0] for j in np.arange(N_0_var)]

            div_Eout_timer = [np.hstack( [div_Eout_timer[j][die_Eout[j] + diff_Eout_eM[j] == 0], div_Eout_timer[j][div_Eout[j] > 0], div_Ein_timer[j][out_Ein[j]*(1-die_Ein[j]) > 0]] ) % trans_steps[3] for j in np.arange(N_0_var)]

            div_Ein_timer = [np.hstack( [div_Ein_timer[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], div_Ein_timer[j][div_Ein[j] > 0], np.zeros(np.sum(diff_pMa_E[j], dtype = int) + (1 - Na_div_flag[j])*np.sum(diff_Na_E[j], dtype = int))] ) % trans_steps[3] for j in np.arange(N_0_var)]
            
            mycEout = [np.hstack( [mycEout[j][die_Eout[j] + diff_Eout_eM[j] == 0], mycEout[j][div_Eout[j] > 0], mycEin[j][out_Ein[j]*(1-die_Ein[j]) > 0]] ) for j in np.arange(N_0_var)]
            
            mycEin = [np.hstack( [mycEin[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], mycEin[j][div_Ein[j] > 0], mycNa[j][diff_Na_E[j] > 0]] )for j in np.arange(N_0_var)]
            
            mycNa = [np.hstack( [mycNa[j], np.zeros(act_N[j], dtype = int), mycNa[j][div_Na[j] == 1]] ) for j in np.arange(N_0_var)]

            cyt_Eout_timer = [np.hstack( [cyt_Eout_timer[j][die_Eout[j] + diff_Eout_eM[j] == 0], cyt_Eout_timer[j][div_Eout[j] > 0], cyt_Ein_timer[j][out_Ein[j]*(1-die_Ein[j]) > 0]] ) for j in np.arange(N_0_var)]
            
            cyt_Ein_timer = [np.hstack( [cyt_Ein_timer[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], cyt_Ein_timer[j][div_Ein[j] > 0], np.zeros(np.sum(diff_pMa_E[j], dtype = int) + (1 - Na_div_flag[j])*np.sum(diff_Na_E[j], dtype = int))] ) for j in np.arange(N_0_var)]
            
            out_Ein_timer = [np.hstack( [out_Ein_timer[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], out_Ein_timer[j][div_Ein[j] > 0], np.zeros(np.sum(diff_pMa_E[j], dtype = int) + (1 - Na_div_flag[j])*np.sum(diff_Na_E[j], dtype = int))] ) for j in np.arange(N_0_var)]

            diff_Eout_eM_timer = [np.hstack( [diff_Eout_eM_timer[j][die_Eout[j] + diff_Eout_eM[j] == 0], diff_Eout_eM_timer[j][div_Eout[j] > 0], diff_Ein_eM_timer[j][out_Ein[j]*(1-die_Ein[j]) > 0]] ) for j in np.arange(N_0_var)]

            diff_Ein_eM_timer = [np.hstack( [diff_Ein_eM_timer[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], diff_Ein_eM_timer[j][div_Ein[j] > 0], np.zeros(np.sum(diff_pMa_E[j], dtype = int) + (1 - Na_div_flag[j])*np.sum(diff_Na_E[j], dtype = int))] ) for j in np.arange(N_0_var)]

            die_Eout_timer = [np.hstack( [die_Eout_timer[j][die_Eout[j] + diff_Eout_eM[j] == 0], die_Eout_timer[j][div_Eout[j] > 0], die_Ein_timer[j][out_Ein[j]*(1-die_Ein[j]) > 0]] ) for j in np.arange(N_0_var)]

            die_Ein_timer = [np.hstack( [die_Ein_timer[j][out_Ein[j] + die_Ein[j] + diff_Ein_eM[j] == 0], die_Ein_timer[j][div_Ein[j] > 0], np.zeros(np.sum(diff_pMa_E[j], dtype = int) + (1 - Na_div_flag[j])*np.sum(diff_Na_E[j], dtype = int))] ) for j in np.arange(N_0_var)]
            
            #### New binding events ####
            b_act_t = p_tcr*Ain[i]/(char_times[0]*Aout_0)
            b_stim_t = [(2 - p_tcr[j])/char_times[1] for j in np.arange(N_0_var)]
            b_IEin_bind = p_tcr*Ain[i]/(char_times[0]*Aout_0)
            b_IEout_bind = 5*d_IE*I[i]/(K_IE + I[i] + Eout_pop)
            b_IN_bind = p_tcr*Ain[i]/(char_times[0]*Aout_0)
            b_IcM_bind = p_tcr*Ain[i]/(char_times[0]*Aout_0)
            b_IpMa_bind = p_tcr*Ain[i]/(char_times[0]*Aout_0)
            
            bound_IN = [np.random.binomial(1, dt*b_IN_bind[j], Na_m[i,j]) if Na_m[i,j] > 0 else 0 for j in np.arange(N_0_var)]
            bound_IcM = [np.random.binomial(1, dt*b_IcM_bind[j], cM_m[i,j]) if cM_m[i,j] > 0 else 0 for j in np.arange(N_0_var)]
            bound_IpMa = [np.random.binomial(1, dt*b_IpMa_bind[j], pMa_m[i,j]) if pMa_m[i,j] > 0 else 0 for j in np.arange(N_0_var)]
                
            bound_IEin = [np.random.binomial(1, dt*b_IEin_bind[j], Ein_m[i,j]) if Ein_m[i,j] > 0 else 0 for j in np.arange(N_0_var)]
            bound_IEout = [np.random.binomial(1, dt*b_IEout_bind, Eout_m[i,j]) if Eout_m[i,j] > 0 else 0 for j in np.arange(N_0_var)]
        
            unbind_Na_timer = [unbind_Na_timer[j] + np.random.binomial(1-unbound_Na[j], dt*b_stim_t[j]*trans_steps[1], 1) if Na_m[i,j] == 1 else 0 for j in np.arange(N_0_var)]
            unbound_Na = [1*(unbind_Na_timer[j] >= trans_steps[1]) if Na_m[i,j] == 1 else 1 for j in np.arange(N_0_var)]
            
            #### MYC Dynamics ####
            mycNa = [mycNa[j] + dt*(b_myc*(1-unbound_Na[j] + np.mean(bound_IN[j])*p_tcr[j]) - (1-p_cyt[j]*H[i])*mycNa[j]*d_myc) if Na_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            
            mycEin = [mycEin[j] + dt*(b_myc*bound_IEin[j]*p_tcr[j] - (1-p_cyt[j]*H[i])*mycEin[j]*d_myc) if Ein_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            
            mycEout = [mycEout[j] + dt*(b_myc*bound_IEout[j]*p_tcr[j] - (1-p_cyt[j]*H[i])*mycEout[j]*d_myc) if Eout_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            
            myccM = [myccM[j] + dt*(b_myc*bound_IcM[j]*p_tcr[j] - (1-p_cyt[j]*H[i])*myccM[j]*d_myc) if cM_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]

            #### transition probabilities modulated by antigen and cytokine signals ####
            p_NaE = [p_XtoY(1-unbound_Na[j], p_cyt[j]*H[i], psi_NE_I, psi_NE_c, F_0 = -1.0, K_I = 0.1, K_H = 0.1, reg_model = reg_model) if Na_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            p_EineM = [p_XtoY(bound_IEin[j]*p_tcr[j], p_cyt[j]*H[i], psi_EeM_I, psi_EeM_c, F_0 = 0.0, K_I = 0.1, K_H = 0.1, reg_model = reg_model) if Ein_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            p_EouteM = [p_XtoY(bound_IEout[j]*p_tcr[j], p_cyt[j]*H[i], psi_EeM_I, psi_EeM_c, F_0 = 0.0, K_I = 0.1, K_H = 0.1, reg_model = reg_model) if Eout_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]
            p_pME = [p_XtoY(bound_IpMa[j]*p_tcr[j], p_cyt[j]*H[i], psi_pME_I, psi_pME_c, F_0 = 1.0, K_I = 0.1, K_H = 0.1, reg_model = reg_model) if pMa_m[i,j] > 0 else np.zeros(0) for j in np.arange(N_0_var)]

            #### Time-dependent rates modulated by antigen and cytokine signals ####
            b_Na_div = [1/char_times[2] for j in np.arange(N_0_var)]
            b_NaE_diff = [p_NaE[j]/(char_times[1] + char_times[2]) for j in np.arange(N_0_var)]
            b_E_div = 0.5*(p_tcr + p_cyt)/char_times[3] #(Ein_m[i] + Eout_m[i] < div_dest)*
            b_cM_div = 0.5*(p_tcr + p_cyt)/char_times[4] #(cM_m[i,:] < np.sqrt(div_dest))*
            b_EineM_diff = [p_EineM[j]/char_times[5] for j in np.arange(N_0_var)]
            b_EouteM_diff =[p_EouteM[j]/char_times[5] for j in np.arange(N_0_var)]
            b_pMa_diff = [2*p_pME[j]/char_times[2] for j in np.arange(N_0_var)]
            b_E_out = 0.5*(p_tcr + p_cyt)/char_times[6] # evidence that this is inversely proportional to stimulation
            d_E_die = [1/char_times[7] for j in np.arange(N_0_var)]
            b_E_cyt = 0.5*(p_tcr + p_cyt)/char_times[8] # rate of T cells becoming cytotoxic
        
            #### Store myc levels ####
            mycNa_m[i] = [np.mean(mycNa[j]) if Na_m[i,j] > 0 else 0.0 for j in np.arange(N_0_var)]
            myccM_m[i] = [np.mean(myccM[j]) if cM_m[i,j] > 0 else 0.0 for j in np.arange(N_0_var)]
            mycEin_m[i] = [np.mean(mycEin[j]) if Ein_m[i,j] > 0 else 0.0 for j in np.arange(N_0_var)]
            mycEout_m[i] = [np.mean(mycEout[j]) if Eout_m[i,j] > 0 else 0.0 for j in np.arange(N_0_var)]
            
            #### Store differentiation probabilities
            p_XE[i] = np.array([np.mean(np.hstack(p_NaE)) if Na_m[i].any() > 0 else 0.0, 
                                np.mean(np.hstack(p_EineM)) if Ein_m[i].any() > 0 else 0.0, 
                                np.mean(np.hstack(p_EouteM)) if Eout_m[i].any() > 0 else 0.0,
                                np.mean(np.hstack(p_pME)) if pMa_m[i].any() > 0 else 0.0])
            
        # Increment time
            t += dt
        
        # Collect population dynamics
        N, Na, cM, E, eM, pM = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(cM_m, axis = 1), np.sum(Ein_m + Eout_m, axis = 1), np.sum(eM_m, axis = 1), np.sum(pM_m, axis = 1)
        
        if k == 0: # primary infection
            dyn_data = np.array([S, I, Ain, Na, E, cM + pM, eM, H])
        elif k == 1: # secondary infection
            dyn_data = np.vstack((dyn_data, np.array([S, I, Ain, N, E, cM + pM, eM, H])))
                                 
    ts = np.linspace(0, duration, steps + 1)
    
    print("This fraction of lineages produced effectors: {}".format(np.sum(1*(np.amax(Ein_m + Eout_m, axis = 0) > 0))/N_0_var))
    print("These lineages produced effector memory: {}".format(np.sum(eM_m[-1,:] > 0)))
    
    return np.array(regulation_coeffs), (dyn_data.T)[(E > E_min) + (I > E_min),:], ts[(E > E_min) + (I > E_min)], (Ein_m + Eout_m)[(E > E_min) + (I > E_min)], p_XE[(E > E_min) + (I > E_min)], mycNa_m[(E > E_min) + (I > E_min)], myccM_m[(E > E_min) + (I > E_min)], mycEin_m[(E > E_min) + (I > E_min)], mycEout_m[(E > E_min) + (I > E_min)]

### (4) Parallelize simulation runs
def sum_sim(I_0 = I_0, b_I = b_I, N_0 = N_0, d_IE = d_IE, d_IH = d_IH,
            regulation_coeffs = psis,
            char_times = [t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_eM_diff, t_E_out, t_E_die, t_E_cyt],
            trans_steps = np.array([1.0, 3.0, 4.0, 4.0, 4.0, 4.0, 2.0, 4.0, 4.0]),
            infection = "prim",
            sim_kind = "agent",
            reg_model = "mwc_like"):
    
    # compute state and costate dynamics
    if sim_kind == "agent":
        rates, dyn, ts, _, _, _, _ = agent_stoch_sim(I_0, b_I, N_0, d_IE, d_IH,
                                        regulation_coeffs = regulation_coeffs,
                                        char_times = char_times,
                                        trans_steps =  trans_steps,
                                        infection = infection,
                                        reg_model = reg_model)
    # extract primary/secondary infection dynamics
        pI, sI, Ain, N, E, cM, eM, H = dyn[:, 1], dyn[:,-7], dyn[:,-6], dyn[:,-5], dyn[:,-4], dyn[:,-3], dyn[:,-2], dyn[:,-1]
        
    else:
        rates, I, N, E, cM, eM, T, ts = stoch_sim(I_0, b_I, tau_I, d_IE, T_I,
                                                  regulation_coeffs = regulation_coeffs,
                                                  char_times = char_times,
                                                  trans_steps = trans_steps,
                                                  infection = infection)
        
    dt = ts[1]-ts[0]
    
    run_data = np.concatenate((regulation_coeffs, [I_0, b_I, N_0, d_IE,
                                                       np.sum( np.log(np.maximum(pI, E_min)) )*dt, 
                                                       np.argmax(sI)*dt,
                                                       np.sum(np.log(np.maximum(sI, E_min)) )*dt,
                                                       np.max(E),
                                                       np.argmax(E)*dt, 
                                                       cM[-1], 
                                                       np.sum(E*dt), 
                                                       np.sum(np.log(np.maximum(E, E_min))*dt),
                                                       eM[-1]]), 
                                  axis = None)
    
    return run_data

stat_names = [r"$\psi_{N,E}^{(I)}$", r"$\psi_{N,E}^{(c)}$", r"$\psi_{E,eM}^{(I)}$", r"$\psi_{E,eM}^{(c)}$", r"$\psi_{pM,E}^{(I)}$", r"$\psi_{pM,E}^{(c)}$", \
              r"$I_0$", r"$b_{I}$", r"$N_0$", r"$d_{I,E}$",\
              r"$\int_0^{T_{sim}} \log(I_{p}) dt$",\
              r"$T_{I}^{max}$", 
              r"$\int_0^{T_{sim}} \log(I_{s}) dt$",\
              r"$E^{max}$",\
              r"$T_{E}^{max}$",\
              r"$(cM)^\infty}$",\
              r"$\int E dt$", \
              r"$\int \log\left(E\right) dt$",\
              r"$(eM)^\infty}$"]

stat_names_for_df = ['psi_NE_c', 'psi_NE_I', 'psi_EeM_c', 'psi_EeM_I', 'psi_pME_c', 'psi_pME_I', 
               'mi_N_0_p_harm', 'mi_N_0_T_max_I', 'mi_N_0_s_harm', 'mi_N_0_max_E','mi_N_0_T_max_E','mi_N_0_inf_cM','mi_N_0_int_E', 'mi_N_0_int_logE', 'mi_N_0_inf_eM',
               'mi_b_I_p_harm', 'mi_b_I_T_max_I', 'mi_b_I_s_harm', 'mi_b_I_max_E','mi_b_I_T_max_E','mi_b_I_inf_cM','mi_b_I_int_E', 'mi_b_I_int_logE', 'mi_b_I_int_eM',
               'p_harm', 'T_max_I', 's_harm', 'max_E','T_max_E','inf_cM','int_E', 'int_logE', 'inf_eM']

# def agent_stoch_sim(I_0 = I_0, b_I = b_I, tau_I = tau_I, d_IE = d_IE, T_I = T_I,
#                     regulation_coeffs = psis,
#                     rates = [ec50_act_mean, b_N_max, b_N_act_max, b_E_max, b_cM_max, b_eM_max, b_c1],
#                     rate_cv = np.array([5.0, 0.2, 0.2, 0.2, 0.5, 0.5, 2.0]),
#                     infection = "prim", 
#                     duration = sim_duration, 
#                     steps = sim_steps):
    
#     dt =  duration/steps
    
#     cov_bE_bM = 0.95
#     # set infection scenario
#     infection_count = 0
#     if infection == "prim":
#         infection_count = 1
#     elif infection == "sec":
#         infection_count = 2
    
#     # draw population of reponding cells for agent-based simulations
#     psi_NE_I, psi_NE_c, psi_cME_I, psi_cME_c, psi_EeM_I, psi_EeM_c = regulation_coeffs
    
#     ec50_act = np.random.lognormal(mean = np.log((rates[0])/np.sqrt(1 + rate_cv[0]**2)), 
#                         sigma = np.sqrt(np.log(1+ rate_cv[0]**2)), size = N_0)
    
#     bN = np.minimum(np.random.lognormal(mean = np.log((rates[1])/np.sqrt(1 + rate_cv[1]**2)), 
#                         sigma = np.sqrt(np.log(1+ rate_cv[1]**2)), size = N_0), 1)
    
#     bN_act = np.minimum(np.random.lognormal(mean = np.log((rates[2])/np.sqrt(1 + rate_cv[2]**2)), 
#                         sigma = np.sqrt(np.log(1+ rate_cv[2]**2)), size = N_0), 4.8)
    
#     bE, bcM = np.exp(np.random.multivariate_normal(mean = [np.log((rates[3])/np.sqrt(1 + rate_cv[3]**2)), np.log((rates[4])/np.sqrt(1 + rate_cv[4]**2))], 
#                                         cov = np.array([[np.log(1+ rate_cv[3]**2), np.sqrt(np.log(1+ rate_cv[3]**2)*np.log(1+ rate_cv[4]**2))*np.log(cov_bE_bM*(np.exp(1)-1)+1)], [np.sqrt(np.log(1+ rate_cv[3]**2)*np.log(1+ rate_cv[4]**2))*np.log(cov_bE_bM*(np.exp(1)-1)+1), np.log(1+ rate_cv[4]**2)]]), size = N_0)).T
#     bE = np.minimum(bE, 4.8)
#     bcM = np.minimum(bcM, 2.5)
    
#     beM = np.minimum(np.random.lognormal(mean = np.log((rates[5])/np.sqrt(1 + rate_cv[5]**2)), 
#                         sigma = np.sqrt(np.log(1+ rate_cv[5]**2)), size = N_0), 2.5)
    
#     bc1 = np.random.lognormal(mean = np.log((rates[6])/np.sqrt(1 + rate_cv[6]**2)), 
#                         sigma = np.sqrt(np.log(1+ rate_cv[6]**2)), size = N_0)
    
#     # draw maximum number of divisions that a cell can sustain: Subramanian et al. (2008)
#     div_dest_prob = [0.0001, 0.0017, 0.0165,0.0826, 0.2206, 0.3151,0.2408,0.0984,0.0215,0.0025, 0.0002]
#     div_dest = 2**(np.random.choice(a = np.arange(12,23), p = div_dest_prob, size = N_0))
    
    
#     for k in np.arange(0, infection_count):
        
#         # define variables for storage
#         if k == 0: # primary infection
#             N_m = np.zeros((steps+1, N_0), dtype = int)
#             N_m[0,:] +=1
#             pM_m = np.zeros((steps+1, N_0), dtype = int)
#         elif k == 1: # secondary infection
#             N_m[0,:] = N_m[-1,:]
#             pM_m[0,:] = cM_m[-1,:] + eM_m[-1,:]
        
#         N_act_m = np.zeros((steps+1, N_0), dtype = int)

#         E_m = np.zeros((steps+1, N_0), dtype = int)
#         cM_m = np.zeros((steps+1, N_0), dtype = int)
#         eM_m = np.zeros((steps+1, N_0), dtype = int)
#         I = np.zeros(steps+1)
#         S = np.zeros(steps+1)
#         eTr = np.zeros(steps+1)
#         p_XE = np.zeros((steps+1, 3))

#         # Run population simulation
#         t = 0.0
#         S[0] = S_0
#         I[0] = I_0

#         for i in np.arange(1, steps + 1):
#             # Compute total population of cell types
#             E_pop, cM_pop, eM_pop = np.sum(E_m[i-1]), np.sum(cM_m[i-1]), np.sum(eM_m[i-1])
#             # check simulation stopping conditions
#             if I[i-1] <= E_min and E_pop <= E_min:
#                 break
            
#             p_t = hl_u(pmhc_per_I*I[i-1], ec50_act,l=l) # antigen activation probability
#             c_t = c1_ss(p_t, tau_I,I[i-1], E_pop, eTr[i-1], bc1) # cytokine dynamics
#             hl_t = hl_u(c_t, k_E) # cytokine activation probability

#             # Run infection dynamics: replication and effector clearance
#             S[i] = S[i-1] + dt*(b_S - d_S*S[i-1] - b_I*S[i-1]*I[i-1])
#             I[i] = I[i-1] + dt*((I[i-1] >= I_0)*np.exp(-t/T_I)*b_I*S[i-1]*I[i-1] - d_IE*I[i-1]*E_pop/(K_IE + I[i-1] + E_pop) - d_I*I[i-1])*(I[i-1] > 0.0)

#             eTr[i] = b_eTr(np.mean(c_t)) - eTr[i-1]*d_eTr(np.mean(c_t))
            
#             # transition probabilities modulated by antigen and cytokine signals
#             p_N_act_E = alpha*((1-psi_NE_I)*(1-p_t) + psi_NE_I*p_t) + (1-alpha)*((1 - psi_NE_c)*(1-hl_t) + psi_NE_c*hl_t)
#             p_E_E = alpha*((1-psi_EeM_I)*(1-p_t) + psi_EeM_I*p_t) + (1-alpha)*((1 - psi_EeM_c)*(1-hl_t) + psi_EeM_c*hl_t)
#             p_M_E = alpha*((1-psi_cME_I)*(1-p_t) + psi_cME_I*p_t) + (1-alpha)*((1 - psi_cME_c)*(1-hl_t) + psi_cME_c*hl_t)
            
#             p_XE[i,:] = np.array([np.mean(p_N_act_E), np.mean(p_M_E), np.mean(p_E_E)])
            
#             # Time-dependent rates modulated by antigen and cytokine signals
#             b_N_t = bN*p_t
#             b_N_act_t = bN_act
#             b_E_t = bE*hl_t*p_E_E*(E_m[i-1,:] < div_dest)
#             b_cM_t = bcM*hl_t*(cM_m[i-1,:] < np.sqrt(div_dest))
#             b_eM_t = b_eM_max*(1-p_E_E)
#             d_E_t = d_E_max*(1-hl_t)

#             # Naive cells have a timer to activation and first division
#             N_act = (np.random.poisson(b_N_t*dt, N_0) > 0)*N_m[i-1,:]
            
#             # Activated naive cells divide and differentiate
#             N_act_diff = np.random.binomial(N_act_m[i-1,:], dt*b_N_act_t, N_0)

#             N_act_to_cM = np.random.binomial(2*N_act_diff, 1 - p_N_act_E, N_0)

#             # New central memory cells divide
#             cM_div = np.random.binomial(cM_m[i-1,:], dt*b_cM_t, N_0)
            
#             # Memory cells from a primary infection divide and differentiate
#             pM_act = np.random.binomial(pM_m[i-1,:], 2*dt*b_N_act_t, N_0)
            
#             pM_to_cM = np.random.binomial(pM_act, 1-p_M_E, N_0)
            
#             # Effector cells divide and differentiate, or die, or both

#             E_div_die_diff = np.random.binomial(E_m[i-1,:], dt*(b_E_t + d_E_t + b_eM_t), N_0)

#             E_div = np.random.binomial(E_div_die_diff, b_E_t/(b_E_t + d_E_t + b_eM_t), N_0)

#             E_die = np.random.binomial(E_div_die_diff-E_div, d_E_t/(d_E_t + b_eM_t), N_0)

#             E_to_eM = E_div_die_diff - E_div - E_die


#             # Update population dynamics
#             N_m[i,:] = N_m[i-1,:] - N_act
#             N_act_m[i,:] = N_act_m[i-1,:] + 2*N_act - N_act_diff
#             pM_m[i,:] = pM_m[i-1,:] - pM_act
#             cM_m[i,:] = cM_m[i-1,:] + N_act_to_cM + cM_div + pM_to_cM # - cM_to_E) 
#             E_m[i,:] = (2*N_act_diff - N_act_to_cM) + (E_m[i-1,:] + E_div - E_to_eM) + (pM_act - pM_to_cM)  - E_die 
#             eM_m[i,:] = eM_m[i-1,:] + E_to_eM

#             # Increment time
#             t += dt
        
#         # Collect population dynamics
#         N, cM, E, eM, pM = np.sum(N_m + N_act_m, axis = 1), np.sum(cM_m, axis = 1), np.sum(E_m, axis = 1), np.sum(eM_m, axis = 1), np.sum(pM_m, axis = 1)
        
#         if k == 0: # primary infection
#             dyn_data = np.array([S, I, N, E, cM + pM, eM, eTr])
#         elif k == 1: # secondary infection
#             dyn_data = np.vstack((dyn_data, np.array([S, I, N, E, cM + pM, eM, eTr])))
                                 
#     ts = np.linspace(0, duration, steps + 1)
    
#     return np.array(regulation_coeffs), (dyn_data.T)[(E > E_min) + (I > E_min),:], ts[(E > E_min) + (I > E_min)], p_XE[(E > E_min) + (I > E_min)]

# ### (4) Parallelize simulation runs
# def sum_sim(I_0 = I_0, b_I = b_I, tau_I = tau_I, d_IE = d_IE, T_I = T_I,
#             regulation_coeffs = psis,
#             rates = [ec50_act_mean, b_N_max, b_N_act_max, b_E_max, b_cM_max, b_eM_max, b_c1],
#             rate_cv = np.array([0.5, 0.2, 0.2, 0.2, 0.5, 0.5, 2.0]),
#             infection = "prim",
#             sim_kind = "agent"):
    
#     # compute state and costate dynamics
#     if sim_kind == "agent":
#         rates, dyn, ts, _ = agent_stoch_sim(I_0, b_I, tau_I, d_IE, T_I,
#                                         regulation_coeffs = regulation_coeffs,
#                                         rates = rates,
#                                         rate_cv =  rate_cv,
#                                         infection = infection)
#     # extract primary/secondary infection dynamics
#         pI, sI, N, E, cM, eM, eTr = dyn[:, 1], dyn[:,-6], dyn[:,-5], dyn[:,-4], dyn[:,-3], dyn[:,-2], dyn[:,-1]
        
#     else:
#         rates, I, N, E, cM, eM, T, ts = stoch_sim(I_0, b_I, tau_I, d_IE, T_I,
#                                                   regulation_coeffs = regulation_coeffs,
#                                                   rates = rates,
#                                                   rate_cv = rate_cv,
#                                                   infection = infection)
        
#     dt = ts[1]-ts[0]
    
#     run_data = np.concatenate((regulation_coeffs, [I_0, b_I, tau_I, d_IE,
#                                                        np.sum( np.log(np.maximum(pI, E_min)) )*dt, 
#                                                        np.argmax(sI)*dt,
#                                                        np.sum(np.log(np.maximum(sI, E_min)) )*dt,
#                                                        np.max(E),
#                                                        np.argmax(E)*dt, 
#                                                        cM[-1], 
#                                                        np.sum(E*dt), 
#                                                        np.sum(np.log(np.maximum(E, E_min))*dt),
#                                                        eM[-1]]), 
#                                   axis = None)
    
#     return run_data

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

### (6) functions for generating sobol sequence grids
def sample_grid(d,m, type = 'discrete'):
    # d = dimension of grid points
    # exponent of size of grid points (power of 2)
    
    if type == 'discrete':
        sample = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
    else:
        sampler = qmc.Sobol(d=d, scramble=False)
        vertices = np.array(list(itertools.product(np.arange(0,3)/2, repeat=d)))
        if m > 0:
            sample = np.vstack((sampler.random_base2(m=m), vertices))
        else:
            sample = vertices
    
    return sample

def sample_pathogen(l_bounds = [0.1*b_I, 0.1*N_0], u_bounds = [10*b_I, 10*N_0], runs = 1000):
    
    sampler = qmc.Sobol(d=2, scramble=False)
    sample = sampler.random_base2(m = int(np.ceil(np.log2(runs))))
    out = qmc.scale(sample, l_bounds, u_bounds)
    
    return out