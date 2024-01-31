import itertools

### Stochastic model dynamics ###
import numpy as np
from scipy.stats import qmc
from scipy import special
from scipy import stats
from math import erf
import numba as nb
import os as os
import pandas as pd
import itertools

### (1) Define simulation parameters
# Define simulation parameters

# infection dynamics
S_0 = 10**7 #susceptible cells
d_S = 0.05
I_0 = 10 # initial detectable levelof infected cells
beta, c, pi = (10**(-7)), 5, 50 # (Chao et al. 2004)
b_I = beta*pi/c # harm per unit virion
d_IE = 12 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE = 10**4 # effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
d_I = np.minimum(10*d_S, S_0*b_I) # successful virus cannot kill cells faster than it infects new ones

# APC dynamics
Aout_0 = 6*(10**4)
d_A = 0.4
b_Ain = 1
delta_Ain = 10 # relative avidity of APCs for T cells compared to infected cells
K_Ain = K_IE/delta_Ain

# Inflammatory response
d_H = 0.5
b_H = 10 # innate response/inflammation per lysed cell compared to natural death
l_H = 1 # cooperativity
K_IH = d_S*S_0/(d_H) # half-max level of instantaneous damage required to trigger innate/inflammatory response
K_EH = K_IH # half-max level of inflammation required to trigger lymphocyte response
ep = 10**(-3) # off-target rate of harm
K_SE = 10**8
kappa = 0.75 # maximal reduction in replication rate due to inflammatory response
d_IH = d_IE*ep
H_0 = 0.0

# Immune cells
N_0 = 100
max_Na = 4
max_expand = 10_000
max_eM_frac = 0.20
t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_EeM_diff, t_E_out, t_E_die, t_E_cyt, t_NaE_diff = 1/2, 3/4, 1/4, 1/3, 1/2, 3.0, 2/3, 3.0, 2/3, 1/2
n_act, n_bind, n_Na_div, n_E_div, n_cM_div, n_EeM_diff, n_E_out, n_E_die, n_E_cyt, n_NaE_diff = 1, 1, 1, 1, 1, 3, 1, 6, 1, 3
E_min = 1 # minimum detectable cell counts

# Division timer
b_myc = 1.0*(10**3)
d_myc = np.log(2)*24/7
myc_thresh = 10**(2.6)

# hyper parameters
alpha = 0.5 # weight of antigenic signals relative to inflamatory signals
zeta = 0.5 # fraction of maximal rate unregulated by signals
psis = np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, 0.5, 0.0]) # regulatory weights: psi_NE_I, psi_NE_H, psi_NE_IH, psi_EeM_I, psi_EeM_H, psi_EeM_IH, psi_pME_I, psi_pME_H, psi_pME_IH
vir_prop = np.vstack((np.array([0, K_SE]),np.array(np.meshgrid(d_S*np.array([2, 5, 10]), K_IE*np.array([1, 5, 10]))).T.reshape(-1,2)))

# define reg options
x_all = np.array(list(itertools.product([-1, 0, 1], repeat =3)))
x_contr = np.vstack((x_all[(np.sum(np.abs(x_all[:,0:2]), axis = 1) == 2)*(x_all[:,-1] == 0)]/2, 
                     x_all[(np.sum(np.abs(x_all), axis = 1) == 1)*(x_all[:,-1] == 0)],
                     x_all[(np.sum(np.abs(x_all[:,0:2]), axis = 1) == 0)*(np.abs(x_all[:,-1]) <= 1)]))

reg_opts = [np.hstack(i) for i in list(itertools.product(x_contr, repeat =3))]

### (2) Define functions for simulations
# Define functions for simulations
@nb.njit
def hl_u(x,k,l=l_H):
    return (x**l)/(k**l + x**l)

def vir(I, d_I, b_I = b_I, model = "dep_harm"):
    if model == "dep_harm":
        out = b_I*d_I*I
        
    elif model == "indep_harm":
        out = b_I*I
    
    return out


def p_XtoY(I_sig, H_sig, psi_I, psi_H, psi_IH, F_0, K_I, K_H, reg_model = "mwc_like"):
    ### variable
    # I_sig := antigenic stimuli
    # H_sig := inflammatory stimuli
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    # reg_model := family of regulatory functions considered: Monod-Wyman-Changeaux inspired, and Hill functions
    
    if reg_model == "mwc_like":
        F_1 = (psi_I*np.log(1 + I_sig/K_I) + psi_H*np.log(1 + H_sig/K_H)) + psi_IH*np.log(1 + (I_sig*H_sig)/(K_I*K_H))
        out = 1/(1 + np.exp(- (F_1 + F_0)))
        
    else:
        out = 0.0
    
    return out


def init_list(length, size):
    
    out = [np.zeros(length, dtype = int) for l in np.arange(size)]
    return out

# functions for generating sobol sequence grids
def sample_grid(d = 2, l_bounds = [d_S, 0.5], u_bounds = [S_0*b_I, 1.0], runs = 1000):
    
    sampler = qmc.Sobol(d=d, scramble=False)
    sample = sampler.random_base2(m = int(np.ceil(np.log2(runs))))
    out = qmc.scale(sample, l_bounds, u_bounds)
    if d == 1:
        out = out.ravel()
    
    return out

#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################
sim_duration = 25
sim_steps = int(0.5*(10**4))

def agent_stoch_sim(S_0 = S_0, I_0 = I_0, b_I = b_I, d_S = d_S, d_I = d_I, d_IE = d_IE, d_IH = d_IH, K_IE = K_IE, K_IH = K_IH, K_SE = K_SE,
                    Aout_0 = Aout_0, b_Ain = b_Ain, b_H = b_H, d_H = d_H, K_EH = K_EH,
                    N_0 = N_0, max_Na = max_Na, b_myc = b_myc, d_myc = d_myc, myc_thresh = myc_thresh, max_expand = max_expand,
                    char_times = [t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_EeM_diff, t_E_out, t_E_die, t_E_cyt, t_NaE_diff],
                    trans_steps = [n_act, n_bind, n_Na_div, n_E_div, n_cM_div, n_EeM_diff, n_E_out, n_E_die, n_E_cyt, n_NaE_diff],
                    regulation_coeffs = psis,
                    alpha = alpha,
                    infection = "prim",
                    vir_model = "dep_harm",
                    duration = sim_duration, 
                    steps = sim_steps):
    
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, d_IH := d_IH, K_IE := K_IE, K_IH := K_IH,
    # Aout_0 := Aout_0, b_Ain := b_Ain, b_H := b_H, d_H := d_H, K_EH := K_EH,
    # N_0 := N_0, max_Na := max_Na, b_myc := b_myc, d_myc := d_myc, myc_thresh := myc_thresh,
    # char_times := [t_act, t_bind, t_Na_div, t_E_div, t_cM_div, t_EeM_diff, t_E_out, t_E_die, t_E_cyt],
    # trans_steps := [n_act, n_bind, n_Na_div, n_E_div, n_cM_div, n_EeM_diff, n_E_out, n_E_die, n_E_cyt])
    
    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_0*d_S
    
    # in case of autoimmune response
    if d_I == 0.0:
        d_Sauto = d_S
        K_SE = K_IE
        I_0 = 0.0
    else:
        d_Sauto = 0.0
    
    # set infection scenario: primary or secondary
    infection_count = 0
    if infection == "prim":
        infection_count = 1
    elif infection == "sec":
        infection_count = 2
    
    # select virulence model:
    if vir_model == "dep_harm":
        b_I = b_I*d_I
    
    # draw population of reponding cells for agent-based simulations
    psi_NE_I, psi_NE_H, psi_NE_IH, psi_EeM_I, psi_EeM_H, psi_EeM_IH, psi_pME_I, psi_pME_H, psi_pME_IH = regulation_coeffs
    
    mu_tcr = 0.8
    var_tcr = mu_tcr*(1-mu_tcr)*0.9
    mu_cyt = 0.8
    var_cyt = mu_cyt*(1-mu_cyt)*0.9
    
    p_tcr = np.ones(N_0_var)#*np.random.beta(((1-mu_tcr)/var_tcr - 1/mu_tcr)*(mu_tcr**2), mu_tcr*(1-mu_tcr)**2/var_tcr + mu_tcr - 1, size = N_0_var) #np.random.uniform(low = 0.75, high = 1.0, size = N_0_var); np.random.beta(mu_tcr**2*((1-mu_tcr)/var_tcr - 1/mu_tcr), mu_tcr*(1-mu_tcr)**2/var_tcr + mu_tcr - 1, size = N_0_var)
    p_cyt = np.ones(N_0_var)#*np.random.beta(((1-mu_cyt)/var_cyt - 1/mu_cyt)*(mu_cyt**2), mu_cyt*(1-mu_cyt)**2/var_cyt + mu_cyt - 1, size = N_0_var) #np.random.uniform(low = 0.75, high = 1.0, size = N_0_var); np.random.beta(mu_cyt**2*((1-mu_cyt)/var_cyt - 1/mu_cyt), mu_cyt*(1-mu_cyt)**2/var_cyt + mu_cyt - 1, size = N_0_var)
    
    for k in np.arange(0, infection_count):
        
        # define variables for storage
        I = np.zeros(int(steps)+1)
        S = np.zeros(int(steps)+1)
        V = np.zeros(int(steps)+1)
        Aout = np.zeros(int(steps)+1)
        Ain = np.zeros(int(steps)+1)
        H = np.zeros(int(steps)+1)
        I_d_I = np.zeros(int(steps)+1)
        I_d_IE = np.zeros(int(steps)+1)
        I_d_S = np.zeros(int(steps)+1)
        
        if k == 0: # primary infection
            N_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
            N_m[0,:] +=1
        elif k == 1: # secondary infection
            N_m[0,:] = 0*N_m[-1,:] # no naive cells during secondary infection
            #print("These lineages did not respond to a primary infection: {}".format(N_m[0,:]))
            cM_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
            cM_m[0,:] = cMa_m[-1,:]
            eM_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
            eM_m[0,:] = eMa_m[-1,:]
            act_cM = [np.zeros(cM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            act_eM = [np.zeros(eM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            #print("These lineages produced memory during the primary infection: {}".format(pM_m[0,:]))
        
        div_cMa_count = np.zeros(N_0_var)
        div_E_count = np.zeros(N_0_var)
        
        Na_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
        cMa_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
        eMa_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
        
        mycNa_m = np.zeros((int(steps)+1, N_0_var))
        myccMa_m = np.zeros((int(steps)+1, N_0_var))
        mycEin_m = np.zeros((int(steps)+1, N_0_var))
        mycEout_m = np.zeros((int(steps)+1, N_0_var))
        if k == 1:
            myccM_m = np.zeros((int(steps)+1, N_0_var))
            myceM_m = np.zeros((int(steps)+1, N_0_var))
        
        Ein_m = np.zeros((int(steps)+1, N_0_var), dtype = int) # effector in lympoid organ
        Eout_m = np.zeros((int(steps)+1, N_0_var), dtype = int) # effector in periphary
        bias_t = np.zeros((int(steps)+1, 5))
        
        # Define event timer variables
        unbind_Na_timer = np.zeros(N_0_var, dtype =int)
        unbound_Na = np.zeros(N_0_var, dtype =int)
        Na_div_flag = np.ones(N_0_var, dtype =int)
        
        div_Na_timer = init_list(0, N_0_var)
        diff_Na_E_timer = init_list(0, N_0_var)
        div_cMa_timer = init_list(0, N_0_var)
        if k == 1:
            div_cM_timer = [np.zeros(cM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            diff_cM_E_timer = [np.zeros(cM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            div_eM_timer = [np.zeros(eM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            diff_eM_E_timer = [np.zeros(eM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            
        div_Ein_timer = init_list(0, N_0_var)
        div_Eout_timer = init_list(0, N_0_var)
        cyt_Ein_timer = init_list(0, N_0_var)
        cyt_Ein = init_list(0, N_0_var)
        cyt_Eout_timer = init_list(0, N_0_var)
        cyt_Eout = init_list(0, N_0_var)
        out_Ein_timer = init_list(0, N_0_var)
        diff_Ein_eM_timer = init_list(0, N_0_var)
        diff_Eout_eM_timer = init_list(0, N_0_var)
        die_Ein_timer = init_list(0, N_0_var)
        die_Eout_timer = init_list(0, N_0_var)
        
        mycNa = init_list(0, N_0_var)
        mycEin = init_list(0, N_0_var)
        mycEout = init_list(0, N_0_var)
        if k == 1:
            myccM = [5*myc_thresh*np.ones(cM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
            myceM = [5*myc_thresh*np.ones(eM_m[0,j], dtype = int) for j in np.arange(N_0_var)]
        myccMa = init_list(0, N_0_var)
        
        p_NaE = np.zeros(N_0_var)
        p_EineM = np.zeros(N_0_var)
        p_EouteM = np.zeros(N_0_var)
        if k == 1:
            p_cME = np.zeros(N_0_var)
            p_eME = np.zeros(N_0_var)
        
        b_unbind_t = np.zeros(N_0_var)
        b_act_t = np.zeros(N_0_var)
        b_Na_div = np.zeros(N_0_var)
        b_NaE_diff = np.zeros(N_0_var)
        b_E_div = np.zeros(N_0_var)
        b_cMa_div = np.zeros(N_0_var)
        b_EineM_diff = np.zeros(N_0_var)
        b_EouteM_diff = np.zeros(N_0_var)
        b_cM_diff = np.zeros(N_0_var)
        b_eM_diff = np.zeros(N_0_var)
        b_E_out = np.zeros(N_0_var)
        d_E_die = np.zeros(N_0_var)
        b_E_cyt = np.zeros(N_0_var)
        
        #################################
        ### RUN POPULATION SIMULATION ###
        #################################
        t = 0.0
        S[0] = S_0
        I[0] = I_0
        V[0] = 0
        Aout[0] = Aout_0
        H[0] = H_0
        
        # errors and troubleshooting
        error_time = 0

        for i in np.arange(1, int(steps) + 1):
            # Compute total population of cell types
            Na_pop, Ein_pop, Eout_pop, cMa_pop, eMa_pop = np.sum(Na_m[i-1]), np.sum(Ein_m[i-1]), np.sum(Eout_m[i-1]), np.sum(cMa_m[i-1]), np.sum(eMa_m[i-1])
            Ein_cyt_pop = np.sum(np.hstack((cyt_Ein))) if Ein_pop > 0 else 0.0
            
            if k == 1:
                cM_pop, eM_pop = np.sum(cM_m[i-1]), np.sum(eM_m[i-1])
                
            #### Run infection dynamics: replication and effector clearance ####
            
            # Update state of susceptible, infected, APCs, and inflammation
            S[i] = S[i-1] + dt*(b_S - (d_S + d_Sauto*np.exp(-t/2))*S[i-1] - (I[i-1] >= I_0)*b_I*I[i-1]*S[i-1]*(1-kappa*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H)) - S[i-1]*(d_IH*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*(Eout_pop)/(K_SE + S[i-1] + Eout_pop)))*(S[i-1] >= 0.0)
            
            I[i] = I[i-1] + dt*((I[i-1] >= I_0)*b_I*I[i-1]*S[i-1]*(1-kappa*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H)) - d_IH*I[i-1]*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) - d_IE*I[i-1]*(Eout_pop)/(K_IE + I[i-1] + Eout_pop) - d_I*I[i-1])*(I[i-1] >= 0.0)
            
            cell_lysed = d_I*I[i-1] + d_IE*I[i-1]*(Eout_pop)/(K_IE + I[i-1] + Eout_pop) + S[i-1]*(d_Sauto*np.exp(-t/2) + d_IE*(Eout_pop)/(K_SE + S[i-1] + Eout_pop))
            H[i] = H[i-1] + dt*(b_H*(cell_lysed) - d_H*H[i-1])*(H[i-1] >= 0.0)
            Aout[i] = Aout[i-1] - b_Ain*Aout[i-1]*H[i-1]**l_H/((K_EH)**l_H + H[i-1]**l_H)*dt*(Aout[i-1] >= 0.0)
            Ain[i] = Ain[i-1] + dt*(b_Ain*Aout[i-1]*H[i-1]**l_H/((K_EH)**l_H + H[i-1]**l_H) - d_A*Ain[i-1] - d_IE*Ain[i-1]*(cMa_pop + Ein_cyt_pop)/(K_IE/delta_Ain + cMa_pop + Ein_cyt_pop))*(Ain[i-1] >= 0.0)
            
            I_d_I[i] = I_d_I[i-1] + dt*(I[i-1] >= I_0)*d_I*I[i-1] + (I[i] if i == int(steps) else 0) # cells killed by infection
            I_d_IE[i] = I_d_IE[i-1] + dt*(I[i-1] >= I_0)*(d_IH*I[i-1]*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*I[i-1]*(Eout_pop)/(K_IE + I[i-1] + Eout_pop)) # cells killed by immune response    
            I_d_S[i] = I_d_S[i-1] + dt*S[i-1]*(d_IH*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*(Eout_pop)/(K_SE + S[i-1] + Eout_pop) + d_Sauto*np.exp(-t/2))
            
            # Set negative values to zero, in 
            if (Ain[i-1] < 0.0 or Ain[i] < 0.0):
                # if error_time == 0:
                #     print("1. Error: Negative APCs {}".format(Ain[i-1]))
                Ain[i-1], Ain[i] = 0.0, 0.0
                error_time = i*dt
                if (Aout[i-1] < 0.0 or Aout[i] < 0.0):
                    Aout[i-1], Aout[i] = 0.0, 0.0
            
            # Iterate over lineages
            for j in np.arange(N_0_var):
                ## I. Recruitment/Priming

                # Phase 1: Naive cells encounter and bind APCs
                act_N = np.random.binomial(N_m[i-1, j], b_act_t[j]*dt, 1) if N_m[i-1, j] > 0 else 0

                # Phase 2: Activated naive cells are bound to APCs and receive stimulation
                # See section with binding times

                # Phase 3: Unbound activated naive cells divide
                div_Na_timer[j] = div_Na_timer[j] + np.random.binomial(unbound_Na[j]*Na_div_flag[j], dt*b_Na_div[j]*trans_steps[2], Na_m[i-1,j]) if Na_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                div_Na = (div_Na_timer[j] >= trans_steps[2])*Na_div_flag[j]

                # After a few quick divisions, activated naive cells can differentiate
                diff_Na_E_timer[j] = (diff_Na_E_timer[j] + np.random.binomial(1, dt*b_NaE_diff[j]*trans_steps[9], Na_m[i-1,j])) if Na_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                diff_Na_E = 1*(diff_Na_E_timer[j] >= trans_steps[9])
                
                ## II. Expansion

                # (a) New central memory cells divide
                div_cMa_timer[j] = div_cMa_timer[j] + np.random.binomial(1, dt*b_cMa_div[j]*trans_steps[4], cMa_m[i-1,j])*(myccMa[j] > myc_thresh) if cMa_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                div_cMa = 1*(div_cMa_timer[j] >= trans_steps[4])*(div_cMa_count[j] < max_expand)

                # (b) Memory cells from a prior infection activate quickly and divide
                if k == 1:
                    act_cM[j] += np.random.binomial(1-act_cM[j], b_act_t[j]*dt, cM_m[i-1,j]) if cM_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    act_eM[j] += np.random.binomial(1-act_eM[j], b_act_t[j]*dt, eM_m[i-1,j]) if eM_m[i-1,j] > 0 else np.zeros(0, dtype = int) 

                    div_cM_timer[j] = div_cM_timer[j] + np.random.binomial(act_cM[j], dt*b_Na_div[j]*trans_steps[2], cM_m[i-1,j]) if cM_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    div_cM = 1*(div_cM_timer[j] >= trans_steps[2])

                    diff_cM_E_timer[j] = diff_cM_E_timer[j] + np.random.binomial(act_cM[j], dt*b_cM_diff[j]*trans_steps[9], cM_m[i-1,j]) if cM_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    diff_cM_E = 1*(diff_cM_E_timer[j] >= trans_steps[9])

                    div_eM_timer[j] = div_eM_timer[j] + np.random.binomial(act_eM[j], dt*b_Na_div[j]*trans_steps[2], eM_m[i-1,j]) if eM_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    div_eM = 1*(div_eM_timer[j] >= trans_steps[2])

                    diff_eM_E_timer[j] = diff_eM_E_timer[j] + np.random.binomial(act_eM[j], dt*b_eM_diff[j]*trans_steps[9], eM_m[i-1,j]) if eM_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    diff_eM_E = 1*(diff_eM_E_timer[j] >= trans_steps[9])

                # (c) Effector cells divide, differentiate, gain cytolytic function, die
                div_Ein_timer[j] = div_Ein_timer[j] + np.random.binomial(1, dt*b_E_div[j]*trans_steps[3], Ein_m[i-1,j])*(mycEin[j] > myc_thresh) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                div_Ein = 1*(div_Ein_timer[j] >= trans_steps[3])*(div_E_count[j] < max_expand)

                if k == 1: # cytolytic function is almost instant in secondary infection
                    cyt_Ein[j] = np.ones(Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                else:
                    cyt_Ein_timer[j] = cyt_Ein_timer[j] + np.random.binomial(1, dt*b_E_cyt[j]*trans_steps[8], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    cyt_Ein[j] = 1*(cyt_Ein_timer[j] >= trans_steps[8])

                out_Ein_timer[j] = out_Ein_timer[j] + np.random.binomial(1, dt*b_E_out[j]*trans_steps[6], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                out_Ein = 1*(out_Ein_timer[j] >= trans_steps[6])

                diff_Ein_eM_timer[j] = diff_Ein_eM_timer[j] + np.random.binomial(1, dt*b_EineM_diff[j]*trans_steps[5], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                diff_Ein_eM = 1*(diff_Ein_eM_timer[j] >= trans_steps[5])

                die_Ein_timer[j] = die_Ein_timer[j] + np.random.binomial(1, dt*d_E_die[j]*trans_steps[7], Ein_m[i-1,j]) if Ein_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                die_Ein = 1*(die_Ein_timer[j] >= trans_steps[7])

                if k == 1: # cytolytic function is almost instant in secondary infection
                    cyt_Eout[j] = np.ones(Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                else:
                    cyt_Eout_timer[j] = cyt_Eout_timer[j] + np.random.binomial(1, dt*b_E_cyt[j]*trans_steps[8], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                    cyt_Eout[j] = 1*(cyt_Eout_timer[j] >= trans_steps[8])

                diff_Eout_eM_timer[j] = diff_Eout_eM_timer[j] + np.random.binomial(1, dt*b_EouteM_diff[j]*trans_steps[5], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                diff_Eout_eM = 1*(diff_Eout_eM_timer[j] >= trans_steps[5])

                div_Eout_timer[j] = div_Eout_timer[j] + np.random.binomial(1, dt*b_E_div[j]*trans_steps[3], Eout_m[i-1,j])* (mycEout[j] > myc_thresh) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                div_Eout = 1*(div_Eout_timer[j] >= trans_steps[3])*(div_E_count[j] < max_expand)

                die_Eout_timer[j] = die_Eout_timer[j] + np.random.binomial(1, dt*d_E_die[j]*trans_steps[7], Eout_m[i-1,j]) if Eout_m[i-1,j] > 0 else np.zeros(0, dtype = int)
                die_Eout = 1*(die_Eout_timer[j] >= trans_steps[7])
                
                #### Update population dynamics: implicit is that differentiation supercedes death if they coincide ####
                N_m[i,j] = N_m[i-1,j] - np.sum(act_N)
                Na_m[i,j] = Na_m[i-1,j] + np.sum(div_Na) + np.sum(act_N) - (1 - Na_div_flag[j])*Na_m[i-1, j]
                if k == 1:
                    cM_m[i,j] = cM_m[i-1,j] - np.sum(div_cM)
                    eM_m[i,j] = eM_m[i-1,j] - np.sum(div_eM)
                cMa_m[i,j] = cMa_m[i-1,j] + np.sum(div_cMa) +  np.sum((1 - diff_Na_E)*(1 - Na_div_flag[j])) + (2*np.sum(div_cM*(1 - diff_cM_E)) if k == 1 else 0)
                Ein_m[i,j] = Ein_m[i-1,j] + np.sum(div_Ein) + np.sum(diff_Na_E)*(1 - Na_div_flag[j]) + ( (2*np.sum(div_cM*(diff_cM_E)) + 2*np.sum(div_eM*(diff_eM_E))) if k == 1 else 0 ) - np.sum(diff_Ein_eM + out_Ein + die_Ein > 0)
                Eout_m[i,j] = Eout_m[i-1,j] + np.sum(div_Eout) + np.sum(out_Ein*(1 - die_Ein)) - np.sum(die_Eout + diff_Eout_eM > 0)
                eMa_m[i,j] = eMa_m[i-1,j] + np.sum(diff_Eout_eM) + np.sum(diff_Ein_eM) + (2*np.sum(div_eM*(1 - diff_eM_E)) if k == 1 else 0)
                
                #### Update and refresh timer variables for division ####
                div_Na_timer[j] = np.hstack( [div_Na_timer[j], np.zeros( int(act_N), dtype = int), div_Na_timer[j][div_Na == 1]] ) % trans_steps[2] if Na_m[i,j] > 0 else np.zeros(0, dtype = int)
            
                diff_Na_E_timer[j] = np.hstack( [diff_Na_E_timer[j], np.zeros( int( act_N), dtype = int), diff_Na_E_timer[j][div_Na == 1]] ) if Na_m[i,j] > 0 else np.zeros(0, dtype = int)
            
                myccMa[j] = np.hstack( [myccMa[j], myccMa[j][div_cMa  > 0], mycNa[j][(1 - diff_Na_E)*(1 - Na_div_flag[j]) > 0], np.tile(myccM[j][div_cM*(1 - diff_cM_E) == 1],2) if k == 1 else np.zeros(0, dtype = int)] )
            
                div_cMa_timer[j] = np.hstack( [div_cMa_timer[j], div_cMa_timer[j][div_cMa > 0], np.zeros( int( ( 2*np.sum(div_cM*(1 - diff_cM_E)) if k == 1 else 0 ) + np.sum((1 - diff_Na_E)*(1 - Na_div_flag[j])) ), dtype = int)] ) % trans_steps[4]
                
                if k == 1: # update only during secondary infections
                    div_cM_timer[j] = div_cM_timer[j][div_cM == 0] if cM_m[i,j] > 0 else np.zeros(0, dtype = int)

                    diff_cM_E_timer[j] = diff_cM_E_timer[j][div_cM == 0] if cM_m[i,j] > 0 else np.zeros(0, dtype = int)

                    div_eM_timer[j] = div_eM_timer[j][div_eM == 0] if eM_m[i,j] > 0 else np.zeros(0, dtype = int)

                    diff_eM_E_timer[j] = diff_eM_E_timer[j][div_eM == 0] if eM_m[i,j] > 0 else np.zeros(0, dtype = int)

                div_Eout_timer[j] = np.hstack( [div_Eout_timer[j][die_Eout + diff_Eout_eM == 0], div_Eout_timer[j][div_Eout > 0], div_Ein_timer[j][out_Ein*(1-die_Ein) > 0]] ) % trans_steps[3]

                div_Ein_timer[j] = np.hstack( [div_Ein_timer[j][out_Ein + die_Ein + diff_Ein_eM == 0], div_Ein_timer[j][div_Ein > 0], np.zeros( int(( (2*np.sum(div_cM*(diff_cM_E)) + 2*np.sum(div_eM*(diff_eM_E))) if k == 1 else 0 ) + (1 - Na_div_flag[j])*np.sum(diff_Na_E)), dtype = int)] ) % trans_steps[3]

                mycEout[j] = np.hstack( [mycEout[j][die_Eout + diff_Eout_eM == 0], mycEout[j][div_Eout > 0], mycEin[j][out_Ein*(1-die_Ein) > 0]] )

                mycEin[j] = np.hstack( [mycEin[j][out_Ein + die_Ein + diff_Ein_eM == 0], mycEin[j][div_Ein > 0], mycNa[j][(1 - Na_div_flag[j])*diff_Na_E > 0], np.tile(myccM[j][div_cM*diff_cM_E == 1],2) if k == 1 else np.zeros(0, dtype = int), np.tile(myceM[j][div_eM*diff_eM_E == 1],2) if k == 1 else np.zeros(0, dtype = int) ] )

                mycNa[j] = np.hstack( [mycNa[j], np.zeros( int(act_N), dtype = int), mycNa[j][div_Na == 1]] )
                
                if k == 1:
                    myccM[j] = (myccM[j]*act_cM[j])[div_cM == 0]
                    
                    act_cM[j] = act_cM[j][div_cM == 0] if cM_m[i,j] > 0 else np.zeros(0, dtype = int)
                    
                    myceM[j] = (myceM[j]*act_eM[j])[div_eM == 0]
                    
                    act_eM[j] = act_eM[j][div_eM == 0] if eM_m[i,j] > 0 else np.zeros(0, dtype = int)
            
                if k == 0:
                    cyt_Eout_timer[j] = np.hstack( [cyt_Eout_timer[j][die_Eout + diff_Eout_eM == 0], cyt_Eout_timer[j][div_Eout > 0], cyt_Ein_timer[j][out_Ein*(1-die_Ein) > 0]] )

                    cyt_Ein_timer[j] = np.hstack( [cyt_Ein_timer[j][out_Ein + die_Ein + diff_Ein_eM == 0], cyt_Ein_timer[j][div_Ein > 0], np.zeros( int((1 - Na_div_flag[j])*np.sum(diff_Na_E)), dtype = int)] )

                out_Ein_timer[j] = np.hstack( [out_Ein_timer[j][out_Ein + die_Ein + diff_Ein_eM == 0], out_Ein_timer[j][div_Ein > 0], np.zeros( int(( (2*np.sum(div_cM*(diff_cM_E)) + 2*np.sum(div_eM*(diff_eM_E))) if k == 1 else 0 ) + (1 - Na_div_flag[j])*np.sum(diff_Na_E)), dtype = int)] )

                diff_Eout_eM_timer[j] = np.hstack( [diff_Eout_eM_timer[j][die_Eout + diff_Eout_eM == 0], diff_Eout_eM_timer[j][div_Eout > 0], diff_Ein_eM_timer[j][out_Ein*(1-die_Ein) > 0]] )

                diff_Ein_eM_timer[j] = np.hstack( [diff_Ein_eM_timer[j][out_Ein + die_Ein + diff_Ein_eM == 0], diff_Ein_eM_timer[j][div_Ein > 0], np.zeros( int(( (2*np.sum(div_cM*(diff_cM_E)) + 2*np.sum(div_eM*(diff_eM_E))) if k == 1 else 0 ) + (1 - Na_div_flag[j])*np.sum(diff_Na_E)), dtype = int)] )

                die_Eout_timer[j] = np.hstack( [die_Eout_timer[j][die_Eout + diff_Eout_eM == 0], die_Eout_timer[j][div_Eout > 0], die_Ein_timer[j][out_Ein*(1-die_Ein) > 0]] )

                die_Ein_timer[j] = np.hstack( [die_Ein_timer[j][out_Ein + die_Ein + diff_Ein_eM == 0], die_Ein_timer[j][div_Ein > 0], np.zeros( int(( (2*np.sum(div_cM*(diff_cM_E)) + 2*np.sum(div_eM*(diff_eM_E))) if k == 1 else 0 ) + (1 - Na_div_flag[j])*np.sum(diff_Na_E)), dtype = int)] )
                
                # Update division flag to allow differentiation to proceed
                Na_div_flag[j] = 1*(Na_m[i,j] < max_Na) if Na_m[i,j] > 0 else 1
                div_cMa_count[j] += np.sum(div_Na) + np.sum(div_cMa) + (np.sum(div_cM) if k == 1 else 0)
                div_E_count[j] += np.sum(div_Na) + ((np.sum(div_cM) + np.sum(div_eM)) if k == 1 else 0) + np.sum(div_Ein) + np.sum(div_Eout)
                
                #### New binding events ####
                b_act_t[j] = p_tcr[j]*Ain[i]/(char_times[0]*(K_IE/delta_Ain + p_tcr[j]*Ain[i] + np.sum(N_m[i-1] + Na_m[i-1] + cMa_m[i] + Ein_m[i])))
                b_unbind_t[j] = np.fmin((K_IE/delta_Ain + np.sum(N_m[i-1] + Na_m[i-1] + cMa_m[i] + Ein_m[i]) + p_tcr[j]*Ain[i])/(p_tcr[j]*char_times[1]*Ain[i]), 1/(dt*trans_steps[1])) if Na_m[i,j] == 1 and Ain[i] >= 1 else 1/dt # set fastest rate of unbinding events
                
                unbind_Na_timer[j] = unbind_Na_timer[j] + np.random.binomial(1-unbound_Na[j], dt*b_unbind_t[j]*trans_steps[1]) if Na_m[i,j] == 1 and Ain[i] >= 1 else 0

                unbound_Na[j] = 1*(unbind_Na_timer[j] >= trans_steps[1]) if Na_m[i,j] == 1 and Ain[i] >= 1 else 1
                
                #### MYC Dynamics ####
                mycNa[j] = mycNa[j] + dt*(b_myc*Ain[i]/(K_IE/(delta_Ain*p_tcr[j]) + Ein_pop + Ain[i]) - (1-H[i-1]**l_H/((K_EH/p_cyt[j])**l_H + H[i-1]**l_H))*mycNa[j]*d_myc) if Na_m[i,j] > 0 else np.zeros(0)

                mycEin[j] = mycEin[j] + dt*(b_myc*Ain[i]/(K_IE/(delta_Ain*p_tcr[j]) + Ein_pop + Ain[i]) - (1-H[i-1]**l_H/((K_EH/p_cyt[j])**l_H + H[i-1]**l_H))*mycEin[j]*d_myc) if Ein_m[i,j] > 0 else np.zeros(0)

                mycEout[j] = mycEout[j] + dt*(b_myc*(I[i] + S[i]*(d_I == 0.0))/(K_IE/p_tcr[j] + Eout_pop + (I[i] + S[i]*(d_I == 0.0))) - (1-H[i-1]**l_H/((K_EH/p_cyt[j])**l_H + H[i-1]**l_H))*mycEout[j]*d_myc) if Eout_m[i,j] > 0 else np.zeros(0)

                myccMa[j] = myccMa[j] + dt*(b_myc*Ain[i]/(K_IE/(delta_Ain*p_tcr[j]) + cMa_pop + Ain[i]) - myccMa[j]*d_myc) if cMa_m[i,j] > 0 else np.zeros(0) # higher decay rate of myc
                
                if k == 1:
                    myccM[j] = myccM[j] + dt*(b_myc*Ain[i]/(K_IE/(delta_Ain*p_tcr[j]) + cM_pop + Ain[i]) - (1-H[i-1]**l_H/((K_EH/p_cyt[j])**l_H + H[i-1]**l_H))*myccM[j]*d_myc) if cM_m[i,j] > 0 else np.zeros(0)

                    myceM[j] = myceM[j] + dt*(b_myc*(I[i] + S[i]*(d_I == 0.0))/(K_IE/p_tcr[j] + eM_pop + (I[i] + S[i]*(d_I == 0.0))) - (1-H[i-1]**l_H/((K_EH/p_cyt[j])**l_H + H[i-1]**l_H))*myceM[j]*d_myc) if eM_m[i,j] > 0 else np.zeros(0)

                #### transition probabilities modulated by antigen and cytokine signals ####
                p_NaE[j] = p_XtoY(p_tcr[j]*(1-unbound_Na[j])*Ain[i], p_cyt[j]*H[i], psi_NE_I, psi_NE_H, psi_NE_IH, F_0 = 0.0, K_I = K_IE/delta_Ain + np.sum(N_m[i] + Na_m[i] + cMa_m[i] + Ein_m[i]), K_H = K_EH) if Na_m[i,j] > 0 else float("nan")
                p_EineM[j] = max_eM_frac*p_XtoY(p_tcr[j]*Ain[i], p_cyt[j]*H[i], psi_EeM_I, psi_EeM_H, psi_EeM_IH, F_0 = 0.0, K_I = K_IE/delta_Ain + np.sum(N_m[i] + Na_m[i] + cMa_m[i] + Ein_m[i]), K_H = K_EH) if Ein_m[i,j] > 0 else float("nan")
                p_EouteM[j] = max_eM_frac*p_XtoY(p_tcr[j]*(I[i] + S[i]*(d_I == 0.0)), p_cyt[j]*H[i], psi_EeM_I, psi_EeM_H, psi_EeM_IH, F_0 = 0.0, K_I = K_IE + np.sum(Eout_m[i] +eMa_m[i]), K_H = K_EH) if Eout_m[i,j] > 0 else float("nan")
                
                if k == 1:
                    p_cME[j] = p_XtoY(p_tcr[j]*Ain[i], p_cyt[j]*H[i], psi_pME_I, psi_pME_H, psi_pME_IH,F_0 = 0.0, K_I = K_IE/delta_Ain + np.sum(N_m[i] + Na_m[i] + cMa_m[i] + Ein_m[i] + cM_m[i]), K_H = K_EH) if cM_m[i,j] > 0 else float("nan")
                    p_eME[j] = p_XtoY(p_tcr[j]*(I[i] + S[i]*(d_I == 0.0)), p_cyt[j]*H[i], psi_pME_I, psi_pME_H, psi_pME_IH, F_0 = 0.0, K_I = K_IE + np.sum(N_m[i] + Na_m[i] + cMa_m[i] + Ein_m[i] + cM_m[i]), K_H = K_EH) if eM_m[i,j] > 0 else float("nan")

                #### Time-dependent rates modulated by antigen and cytokine signals ####
                b_Na_div[j] = 1/char_times[2]
                b_NaE_diff[j] = np.fmin(1/(trans_steps[9]*dt), p_NaE[j]**(1/trans_steps[9])/(1 - p_NaE[j]**(1/trans_steps[9]))*1/((char_times[1] + 2*char_times[2])*trans_steps[9])) if Na_m[i,j] > 0 else 0.0
                b_E_div[j] = (alpha*p_tcr[j] + (1-alpha)*p_cyt[j])/char_times[3]
                b_cMa_div[j] = (alpha*p_tcr[j] + (1-alpha)*p_cyt[j])/char_times[4]
                b_EineM_diff[j] = np.fmin(1/(trans_steps[5]*dt), p_EineM[j]**(1/trans_steps[5])/(1 - p_EineM[j]**(1/trans_steps[5]))*1/(trans_steps[5]*char_times[7])) if Ein_m[i,j] > 0 else 0.0
                b_EouteM_diff[j] = np.fmin(1/(trans_steps[5]*dt), p_EouteM[j]**(1/trans_steps[5])/(1 - p_EouteM[j]**(1/trans_steps[5]))*1/(trans_steps[5]*char_times[7])) if Eout_m[i,j] > 0 else 0.0
                
                if k == 1:
                    b_cM_diff[j] = np.fmin(1/(trans_steps[9]*dt), p_cME[j]**(1/trans_steps[9])/(1 - p_cME[j]**(1/trans_steps[9]))/(char_times[2]*trans_steps[9])) if p_cME[j]*cM_m[i,j] > 0 else 0.0
                    b_eM_diff[j] = np.fmin(1/(trans_steps[9]*dt), p_eME[j]**(1/trans_steps[9])/(1 - p_eME[j]**(1/trans_steps[9]))/(char_times[2]*trans_steps[9])) if p_eME[j]*eM_m[i,j] > 0 else 0.0
                    
                b_E_out[j] = (alpha*p_tcr[j] + (1-alpha)*p_cyt[j])/char_times[6] # evidence that this is inversely proportional to stimulation
                d_E_die[j] = 1/char_times[7]
                b_E_cyt[j] = (alpha*p_tcr[j] + (1-alpha)*p_cyt[j])/char_times[8]# rate of T cells becoming cytotoxic

                #### Store myc levels ####
                mycNa_m[i,j] = np.mean(mycNa[j]) if Na_m[i,j] > 0 else 0.0
                myccMa_m[i,j] = np.mean(myccMa[j]) if cMa_m[i,j] > 0 else 0.0
                mycEin_m[i,j] = np.mean(mycEin[j]) if Ein_m[i,j] > 0 else 0.0
                mycEout_m[i,j] = np.mean(mycEout[j]) if Eout_m[i,j] > 0 else 0.0
                if k == 1:
                    myccM_m[i,j] = np.mean(myccM[j]) if cM_m[i,j] > 0 else 0.0
                    myceM_m[i,j] = np.mean(myceM[j]) if eM_m[i,j] > 0 else 0.0
            
            #### Store differentiation biases
            bias_t[i] = np.array([np.nanmean(np.hstack(p_NaE)) if Na_m[i].any() > 0 else 0.0, 
                                np.nanmean(np.hstack(p_EineM)) if Ein_m[i].any() > 0 else 0.0, 
                                np.nanmean(np.hstack(p_EouteM)) if Eout_m[i].any() > 0 else 0.0,
                                np.nanmean(np.hstack(p_cME)) if k == 1 and cM_m[i].any() > 0 else 0.0,
                                np.nanmean(np.hstack(p_eME)) if k == 1 and eM_m[i].any() > 0 else 0.0])
            
        # Increment time
            t += dt
        
        # Collect population dynamics
        N, Na, cMa, E, eMa = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(cMa_m, axis = 1), np.sum(Ein_m + Eout_m, axis = 1), np.sum(eMa_m, axis = 1)
        if k == 1:
            cM, eM = np.sum(cM_m, axis = 1), np.sum(eM_m, axis = 1)
    
        lineage_comp = np.vstack([np.amax(cMa_m + N_m if k == 0 else cM_m + cMa_m, axis = 0) ,
                                  np.amax(Ein_m + Eout_m, axis = 0),
                                  np.amax(eMa_m if k == 0 else eM_m + eMa_m, axis = 0),
                                  p_tcr,
                                  p_cyt])
        
        if k == 0: # primary infection
            dyn_data = np.array([S, I, Ain, Na, E, cMa, eMa, H, I_d_I + I_d_IE, I_d_S]).T
            prim_bias = bias_t
        elif k == 1: # secondary infection
            dyn_data = np.hstack((dyn_data, np.array([S, I, Ain, Na + cM + eM, E, cMa, eMa, H, I_d_I+ I_d_IE, I_d_S]).T ))
            sec_bias = bias_t
                                 
    ts = np.linspace(0, duration, int(steps) + 1)

    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, sS, pI, sI, Ain, N, pE, sE, pcM, scM, peM, seM, pH, sH, pI_d_I, sI_d_I, pI_d_S, sI_d_S = dyn_data[:,0], dyn_data[:,-10], dyn_data[:,1], dyn_data[:,-9], dyn_data[:,-8], dyn_data[:,3], dyn_data[:, 4], dyn_data[:,-6], dyn_data[:,5], dyn_data[:, -5], dyn_data[:,6], dyn_data[:,-4], dyn_data[:,7], dyn_data[:,-3], dyn_data[:,8], dyn_data[:,-2], dyn_data[:,9], dyn_data[:,-1]
        
    dt = ts[1]-ts[0]
    
    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, d_IH, K_IE, K_IH,
              Aout_0, b_Ain, b_H, d_H, K_EH,
              N_0, max_Na, b_myc, d_myc, myc_thresh]),
              char_times,
              trans_steps,
              regulation_coeffs))

    sim_summary = np.concatenate((parameters,
                      np.array([np.sum(pI*dt), 
                       np.sum(sI*dt),
                       np.argmax(pI)*dt,
                       np.argmax(sI)*dt,
                       np.amax(pI_d_I), 
                       np.amax(sI_d_I), 
                       np.amax(pI_d_S),
                       np.amax(sI_d_S),
                       np.max(pE),
                       np.max(sE),
                       np.argmax(pE)*dt,
                       np.argmax(sE)*dt,
                       np.max(pcM),
                       scM[-1], 
                       np.sum(pE*dt), 
                       np.sum(sE*dt),
                       np.max(peM),
                       seM[-1],
                       np.sum(pH*dt),
                       np.sum(sH*dt),
                       np.amin(pS),
                       np.amin(sS),
                       stats.spearmanr(pI, pH).statistic if np.var(pI)*np.var(pH) > 0.0 else 0.0,
                       stats.spearmanr(sI, sH).statistic if np.var(sI)*np.var(sH) > 0.0 else 0.0],
                       )))

    out_dict = {"reg_coeffs": np.array(regulation_coeffs), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "sec_diff_bias": sec_bias if k == 1 else [],"eff_by_lin": (Ein_m + Eout_m), "Na_myc_by_lin": myccM_m if k == 1 else mycNa_m, "cMa_myc_by_lin":myccMa_m, "Ein_myc_by_lin": mycEin_m, "Eout_myc_by_lin": mycEout_m, "parameters": parameters, "sumary_stats": sim_summary}
    
    return out_dict


stat_names = [r"$\psi_{N,E}^{(I)}$", 
              r"$\psi_{N,E}^{(H)}$",
              r"$\psi_{N,E}^{(I,H)}$",
              r"$\psi_{E,eM}^{(I)}$", 
              r"$\psi_{E,eM}^{(H)}$",
              r"$\psi_{E,eM}^{(I,H)}$", 
              r"$\psi_{pM,E}^{(I)}$", 
              r"$\psi_{pM,E}^{(H)}$",
              r"$\psi_{pM,E}^{(I,H)}$",
              r"$\int_0^{T_{sim}} I_{p}dt$",
              r"$\int_0^{T_{sim}} I_{s}dt$",
              r"$T_{I_p}^{max}$",
              r"$T_{I_s}^{max}$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{p}dt$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{s}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{p}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{s}dt$",
              r"$E_p^{max}$",
              r"$E_s^{max}$",
              r"$T_{E_p}^{max}$",
              r"$T_{E_s}^{max}$",
              r"$(cM_p)^\infty$",
              r"$(cM_s)^\infty$",
              r"$\int E_p dt$",
              r"$\int E_s dt$",
              r"$(eM_p)^\infty$",
              r"$(eM_s)^\infty$",
              r"$\int H_p dt$",
              r"$\int H_s dt$",
              r"$S_p^{min}$",
              r"$S_s^{min}$",
              r"$C(I_p,H_p)$",
              r"$C(I_s,H_s)$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$A_{out}^{(0)}$", r"$b_{A_in}$", r"$b_H$", r"$d_H$", r"$K_{H,E}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N^*,A_{in}}^{(+)}$", r"$\tau_{N^*,A_{in}}^{(-)}$", r"$\tau_{N^*}$", r"$\tau_E$", r"$\tau_{cM}$", r"$\tau_{eM}$", r"$\tau_{E_{out}}$", r"$\tau_{E_{die}}$", r"$\tau_{E_{cyt}}$",
               r"$n_{N^*,A_{in}}^{(+)}$", r"$n_{N^*,A_{in}}^{(-)}$", r"$n_{N^*}$", r"$n_E$", r"$n_{cM}$", r"$n_{eM}$", r"$n_{E_{out}}$", r"$n_{E_{die}}$", r"$n_{E_{cyt}}$",
               r"$\psi_{N,E}^{(I)}$", r"$\psi_{N,E}^{(H)}$", r"$\psi_{N,E}^{(I,H)}$", r"$\psi_{E,eM}^{(I)}$", r"$\psi_{E,eM}^{(H)}$", r"$\psi_{E,eM}^{(I,H)}$", r"$\psi_{pM,E}^{(I)}$", r"$\psi_{pM,E}^{(H)}$", r"$\psi_{pM,E}^{(I,H)}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'd_IH', 'K_IE',
                      'K_IH', 'Aout_0', 'b_Ain', 'b_H', 'd_H', 'K_EH',
                      'N_0', 'max_Na', 'b_myc', 'd_myc', 'myc_thresh',
                      't_act', 't_bind', 't_Na_div', 't_E_div', 't_cM_div', 
                      't_EeM_diff', 't_E_out', 't_E_die', 't_E_cyt', 't_NaE_diff',
                      'n_act', 'n_bind', 'n_Na_div', 'n_E_div', 'n_cM_div',
                      'n_EeM_diff', 'n_E_out', 'n_E_die', 'n_E_cyt','n_NaE_diff',
                      'psi_NE_I', 'psi_NE_H', 'psi_NE_IH','psi_EeM_I', 'psi_EeM_H', 'psi_EeM_IH','psi_pME_I', 'psi_pME_H', 'psi_pME_IH']

stat_names_for_df = ['p_load', 's_load','T_max_pI', 'T_max_sI', 'harm_pI', 'harm_sI', 
                     'harm_pS', 'harm_sS', 'max_pE', 'max_sE','T_max_pE', 'T_max_sE', 
                     'inf_pcM', 'inf_scM', 'int_pE', 'int_sE','inf_peM', 'inf_seM',
                     'int_pH', 'int_sH', 'min_pS', 'min_sS', 'corr_pSig', 'corr_sSig']

### (4) define basic mutual information function
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression

def fast_mi(contingency):
    """
    This comes from sklearn.metrics.mutual_info_score and takes only the
    piece of code relevant for contingency being a numpy array.
    """
    from math import log
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)

def calc_MI(x, y, bin_num = 50, correction = False, seed=None, replicates = 20):
    rng = np.random.default_rng(seed)
    
    subsample_size = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 1.0])*x.size

    mi_data = np.zeros((subsample_size.size * replicates, 2))
    entry = 0
    
    for sub in subsample_size:
        # Precompute permutations of indices [0, int(sub) - 1] to be used
        # for shuffling data.
        permutations = rng.permuted(np.tile(np.arange(int(sub)), replicates)
                                    .reshape(replicates, int(sub)),
                                    axis=1)
        
        # Sample integers between [0, x.size - 1] with replacement
        # for all replicates.
        choices = rng.integers(low=0, high=x.size, size=(replicates, int(sub)))
        
        # Compute the quantiles for each replicate in a vectorized manner.
        bx = np.quantile(x[choices], np.linspace(0, 1, bin_num + 1), axis=1).T
        by = np.quantile(y[choices], np.linspace(0, 1, bin_num + 1), axis=1).T

        for i in np.arange(0,replicates):
            # Quantiles can be degenerate if there's not enough data.
            unique_bx, unique_by = np.unique(bx[i]), np.unique(by[i])
            
            if unique_bx.size == 1:
                unique_bx = np.append(unique_bx, unique_bx + 1)

            if unique_by.size == 1:
                unique_by = np.append(unique_by, unique_by + 1)
                
            c_xy = np.histogram2d(x[choices[i]], y[choices[i]], (unique_bx, unique_by))[0]
            mi_raw = fast_mi(c_xy) / np.log(2)
            
            c_xy_shuffle = np.histogram2d(x[choices[i]], y[choices[i]][permutations[i]], (unique_bx, unique_by))[0]
            mi_correction = fast_mi(c_xy_shuffle) / np.log(2)

            if correction == True:
                # MI correction by shuffling data
                out = mi_raw - mi_correction
            else:
                out = mi_raw

            mi_data[entry, 1], mi_data[entry, 0] = out, 1/sub

            entry += 1

    lr = LinearRegression()
    lr.fit(mi_data[:,0].reshape(-1,1), mi_data[:,1].reshape(-1,1))

    return lr.intercept_

def reg_score(psi_I, psi_H, reg_logic):
    
    if reg_logic == 0:
        if psi_I == 0:
            out = psi_H
        elif psi_H == 0:
            out = psi_I
        else:
            out = np.minimum(psi_I,psi_H)
            
    if reg_logic == 1:
        if psi_I == 0:
            out = psi_H
        elif psi_H == 0:
            out = psi_I
        else:
            out = (psi_I + psi_H)/2
            
    return out