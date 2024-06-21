import itertools

### Stochastic model dynamics ###
import numpy as np
from scipy.stats import qmc
from scipy import special
from scipy import stats
from scipy.stats import gamma
from math import erf
import numba as nb
import os as os
import pandas as pd
import itertools

### (1) Define simulation parameters
# Define simulation parameters

# infection dynamics
S_0 = 10**7 #susceptible cells
d_S = 0.01
b_I = (10**(-7)) # harm per unit virion (Chao et al. 2004, Iwami et al. 2015)
I_0 = 1000 # initial detectable levelof infected cells
d_IE = 12 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE_min = 10**4
K_IE = 10**4 # effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
K_EI = K_IE
d_I = np.minimum(d_S, S_0*b_I) # successful virus cannot kill cells faster than it infects new ones

# APC dynamics
A_init = (10**3)
b_Aout = 10.0
b_Ain = 1.0

# Inflammatory response
d_H = 2.0
b_H = 1 # innate response/inflammation per lysed cell compared to natural death
l_H = 1 # cooperativity
K_IH = b_H*K_IE_min # half-max level of instantaneous damage required to trigger innate/inflammatory response
K_EH = 1*K_IH # half-max level of inflammation required to trigger lymphocyte response
ep = 0*10**(-3) # off-target rate of harm
K_SE = 10*S_0
kappa = 0.0 # maximal reduction in replication rate due to inflammatory response
d_IH = d_IE*ep
H_0 = 0.0

# Immune cells
N_0 = 100
max_Na = 2**3
max_expand = 2**16 #(Marchingo et al.)
t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die, t_E_cyt = 1.0, 3/4, 1/4, 1/3, 1/2, 15.0, 2.0, 2/3
rel_NE_to_M_diff = 5
rel_persist_M = 5 # d_eM/d_cM

# Division timer
d_myc = np.log(2)*24/7 # *np.log(2)
myc_thresh = 10**(2.6)
b_myc = 3*myc_thresh/t_bind

# hyper parameters
alpha = 0.5 # weight of antigenic signals relative to inflamatory signals
vir_prop = np.array(np.meshgrid(d_S*np.logspace(np.log10(20), np.log10(50), 2), # vary d_I
                                   K_IE*np.logspace(0.0, 2.0, 2), # vary K_IE
                                   b_I*np.logspace(np.log10(0.75), 0.0, 2) # vary b_I
                                           )).T.reshape(-1,3)  # vary K_EH

# define reg options
psi_max = 4.0
xv, yv = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
grid_2d = psi_max*np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],[np.sin(np.pi/4), np.cos(np.pi/4)]]).dot(np.vstack([xv.ravel(), yv.ravel()]))/np.sqrt(2)
psi_2d = np.vstack([np.array([[x[0], x[1], psi_max - np.abs(x[0]) - np.abs(x[1])], [x[0], x[1], -(psi_max - np.abs(x[0]) - np.abs(x[1]))]]) for x in grid_2d.T])
#psi_opts = np.array(list(itertools.product(psi_2d.tolist(), repeat = 4))).reshape(-1,12)

NM_psis = [0.0, 0.0, 0.0] # regulatory weights: psi_M_I, psi_M_H, psi_M_IH
EM_psis = [0.0, 0.0, 0.0] #
act_psis = [0.0, 0.0, 0.0] #[psi_max/4, psi_max/4, psi_max/2]
exp_psis = [0.0, 0.0, 0.0] #[-psi_max/2, -psi_max/2, 0.0]
F_0s = np.array([0.0, 0.0, 0.0, 0.0])

### (2) Define functions for simulations
# Define functions for simulations
def f_XtoY(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, psi_1_2 = 0.0, psi_1_3 = 0.0, psi_2_3 = 0.0, F_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0, reg_model = "mwc_like"):
    ### variable
    # sig_1 := antigenic stimuli
    # sig_2 := inflammatory stimuli
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    # reg_model := family of regulatory functions considered: Monod-Wyman-Changeaux inspired, and Hill functions
    
    if reg_model == "mwc_like":
        F_1 = psi_1*np.log((1 + sig_1/K_1)/2) + psi_2*np.log((1 + sig_2/K_2)/2) + psi_3*np.log((1 + sig_3/K_3)/2) + psi_1_2*np.log((1 + (sig_1*sig_2)/(K_1*K_2))/2) + psi_1_3*np.log((1 + (sig_1*sig_3)/(K_1*K_3))/2) + psi_2_3*np.log((1 + (sig_2*sig_3)/(K_2*K_3))/2)
        out = 1/(1 + np.exp(- (F_1 + F_0)))
        
    else:
        out = 0.0
    
    return out

#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################
sim_duration = 20
sim_steps = int(0.5*(10**4))

def lin_stoch_sim(S_0 = S_0, I_0 = I_0, b_I = b_I, d_S = d_S, d_I = d_I, d_IE = d_IE, d_IH = d_IH, K_IE = K_IE, K_IH = K_IH, K_SE = K_SE,
                    A_init = A_init, b_Ain = b_Ain, b_H = b_H, d_H = d_H, K_EI = K_EI, K_EH = K_EH, kappa = kappa,
                    N_0 = N_0, max_Na = max_Na, b_myc = b_myc, d_myc = d_myc, myc_thresh = myc_thresh, max_expand = max_expand,
                    char_times = [t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die, t_E_cyt],
                    NM_regulation = NM_psis,
                    EM_regulation = EM_psis,
                    activation_regulation = act_psis,
                    expansion_regulation = exp_psis,
                    regulation_bias = F_0s,
                    alpha = alpha,
                    infection = "prim",
                    vir_model = "dep_harm",
                    duration = sim_duration, 
                    steps = sim_steps,
                    out_data = "small"):
    
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, d_IH := d_IH, K_IE := K_IE, K_IH := K_IH,
    # A_init := A_init, b_Ain := b_Ain, b_H := b_H, d_H := d_H, K_EI := K_EI, K_EH := K_EH,
    # N_0 := N_0, max_Na := max_Na, b_myc := b_myc, d_myc := d_myc, myc_thresh := myc_thresh,
    # char_times := [t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die],
    
    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_0*d_S
    
    # in case of autoimmune response
    if d_I == 0.0:
        d_Sauto = 1.0*d_S 
        K_SE = K_IE
        
        I_0 = 0.0
    else:
        d_Sauto = 0.0

    t_Hauto = 0.25 # duration of autoimmune inflammation
    
    # set infection scenario: primary or secondary
    infection_count = 0
    if infection == "prim":
        infection_count = 1
    elif infection == "sec":
        infection_count = 2
    
    # draw population of reponding cells for agent-based simulations
    psi_Nact_I, psi_Nact_H, psi_Nact_IH = activation_regulation[0], activation_regulation[1], activation_regulation[2]/2
    psi_NM_I, psi_NM_H, psi_NM_IH = NM_regulation[0], NM_regulation[1], NM_regulation[2]/2
    psi_EM_I, psi_EM_H, psi_EM_IH = EM_regulation[0], EM_regulation[1], EM_regulation[2]/2
    psi_Ediv_I, psi_Ediv_H, psi_Ediv_IH = expansion_regulation[0], expansion_regulation[1], expansion_regulation[2]/2
    
    F0_Nact, F0_NM, F0_EM, F0_Ediv = -0*np.sum(activation_regulation)*np.log(2), -0*np.sum(NM_regulation)*np.log(2), -0*np.sum(EM_regulation)*np.log(2), -0*np.sum(expansion_regulation)*np.log(2)
    
    p_tcr = 1.0 #np.ones(N_0_var)
    p_cyt = 1.0 #np.ones(N_0_var)
    
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
            N_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32)
            N_m[0,:] +=1
            T_Ecyt = [[] for i in np.arange(0, N_0_var)]
        elif k == 1: # secondary infection
            N_m[0,:] = 0*N_m[-1,:] # no naive cells during secondary infection
            T_sEcytM = T_pEcytM[M_survive > 0]
            M_m = np.zeros((int(steps)+1, pM_count), dtype = np.int32)
            M_m[0,:] += 1
            act_M = np.zeros(pM_count, dtype = np.int32)

        div_E_count = np.zeros(N_0_var if k == 0 else pM_count)
        diff_EpM_count = np.zeros(N_0_var if k == 0 else pM_count, dtype = np.int32)
        
        Na_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32)
        Ma_m = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count), dtype = np.int32)
        E_m = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count), dtype = np.int32) # effector in periphary
        T_E = np.zeros(N_0_var if k == 0 else pM_count)
        
        if out_data == "full":
            mycNa_m = np.zeros((int(steps)+1, N_0_var))
            mycMa_m = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count))
            mycE_m = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count))
            if k == 1:
                mycM_m = np.zeros((int(steps)+1, pM_count))
        
        bias_t = np.zeros((int(steps)+1, 5))
        
        # Define event timer variables
        unbound_Na = np.zeros(N_0_var, dtype =np.int32)
        Na_div_flag = np.ones(N_0_var, dtype =np.int32)
        
        mycNa = np.zeros(N_0_var)
        mycE = np.zeros(N_0_var if k == 0 else pM_count)
        mycMa = np.zeros(N_0_var if k == 0 else pM_count)
        if k == 1:
            mycM = 4*myc_thresh*np.ones(pM_count)
        
        p_NaM = np.zeros((int(steps)+1, N_0_var))
        p_EM = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count))
        p_Nact = np.zeros((int(steps)+1, N_0_var))
        p_Ediv = np.zeros((int(steps)+1, N_0_var if k == 0 else pM_count))
        if k == 1:
            p_MM = np.zeros((int(steps)+1, pM_count))
        
        b_unbind_t = np.zeros(N_0_var)
        b_N_act = np.zeros(N_0_var)
        b_Na_div = np.ones(N_0_var)/char_times[2]
        b_E_div = np.ones(N_0_var if k == 0 else pM_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
        b_Ma_div = np.ones(N_0_var if k == 0 else pM_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]
        d_E_die = np.ones(N_0_var if k == 0 else pM_count)/char_times[6]
        if k == 1:
            b_M_act = np.zeros(pM_count)
            b_M_div = np.ones(pM_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[2]   
        
        #################################
        ### RUN POPULATION SIMULATION ###
        #################################
        t = 0.0
        S[0] = S_0
        I[0] = I_0
        V[0] = 0
        Aout[0] = b_Aout/(b_Aout + b_Ain)*A_init
        Ain[0] = b_Ain/(b_Aout + b_Ain)*A_init
        H[0] = H_0

        for i in np.arange(1, int(steps) + 1):
            # select virulence model:
            if vir_model == "dep_harm": # makes  average virus produced roughly the same independent of infected death rate
                b_I_t = b_I + d_I/S_0
            else:
                b_I_t = b_I  
        
            # Compute total population of cell types
            Na_pop, E_pop, Ma_pop = np.sum(Na_m[i-1]), np.sum(E_m[i-1]), np.sum(Ma_m[i-1])
            
            if k == 1:
                M_pop = np.sum(M_m[i-1])
                
            #### Run infection dynamics: replication and effector clearance ####
            
            # Update state of susceptible, infected, APCs, and inflammation
            S[i] = S[i-1] + dt*(b_S - (d_S + d_Sauto*np.exp(-((t-3.0)/t_Hauto)**2))*S[i-1] - b_I_t*I[i-1]*S[i-1]*(I[i-1] >= I_0) - S[i-1]*d_IE*(E_pop)/(K_SE + S[i-1] + E_pop))*(S[i-1] >= 1.0)
            
            I[i] = I[i-1] + dt*(b_I_t*I[i-1]*S[i-1]*(I[i-1] >= I_0) - d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop) - (d_I)*I[i-1])*(I[i-1] >= 1.0)
            
            cell_lysed = d_I*I[i-1] + 0*d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop) + S[i-1]*(d_Sauto*np.exp(-((t-2.5)/t_Hauto)**2) + 0*d_IE*(E_pop)/(K_SE + S[i-1] + E_pop))
            
            H[i] = H[i-1] + dt*(b_H*(cell_lysed) - d_H*H[i-1])*(H[i-1] >= 0.0)
            
            Aout[i] = Aout[i-1] + dt*(b_Aout*Ain[i-1]*f_XtoY(sig_2 = p_cyt*H[i-1], psi_2 = -psi_max, F_0 = -0.0, K_2 = K_EH) - b_Ain*Aout[i-1])*(Aout[i-1] >= 0.0)
            
            Ain[i] = Ain[i-1] + dt*(b_Ain*Aout[i-1] - b_Aout*Ain[i-1]*f_XtoY(sig_2 = p_cyt*H[i-1], psi_2 = -psi_max, F_0 = -0.0, K_2 = K_EH) - d_IE*Ain[i-1]*Na_pop/(K_IE + Na_pop))*(Ain[i-1] >= 0.0)
            
            I_d_I[i] = I_d_I[i-1] + dt*d_I*I[i-1] + (I[i] if i == int(steps) else 0) # cells killed by infection
            
            I_d_IE[i] = I_d_IE[i-1] + dt*d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop) # cells killed by immune response
            
            I_d_S[i] = I_d_S[i-1] + dt*S[i-1]*(d_IE*(E_pop)/(K_SE + S[i-1] + E_pop) + d_Sauto*np.exp(-((t-2.5)/t_Hauto)**2))
            
            ## I. Recruitment/Priming

            # (a) Phase 1: Naive cells encounter and bind APCs
            act_N = np.random.binomial(N_m[i-1], b_N_act*dt)

            # (a) Phase 2: Activated naive cells are bound to APCs and receive stimulation.

            # (a) Phase 3: Unbound activated naive cells divide and then differentiate
            div_Na = np.random.binomial(Na_m[i-1]*unbound_Na*Na_div_flag, dt*b_Na_div)

            diff_NaM = np.random.binomial(Na_m[i-1]*(1-Na_div_flag)*unbound_Na, 1 - np.exp(-np.sum(p_NaM[0:i-1], axis = 0)) if np.sum(Na_m[i-1]) > 0 else 0)
            
            # (b) Memory cells from a prior infection activate quickly and divide
            if k == 1:
                act_M += np.random.binomial(M_m[i-1] - act_M, b_M_act*dt)
                div_M = np.random.binomial(act_M, dt*b_M_div)
                diff_MM = np.random.binomial(2*div_M, 1 - np.exp(-np.sum(p_MM[np.maximum(0,i-1-int(char_times[2]/dt)):i-1])), axis = 0)
            
            ## II. Expansion

            # (a) New central memory cells divide
            div_Ma = np.random.binomial(Ma_m[i-1] - diff_EpM_count, dt*b_Ma_div)

            # (b) Effector cells divide, differentiate, die
            die_E = np.random.binomial(E_m[i-1], d_E_die*dt)
            div_E = np.random.binomial(E_m[i-1] - die_E, dt*(b_E_div))
            diff_EM = np.random.binomial(E_m[i-1] - die_E + div_E, p_EM[i-1])
            
            #### Update population dynamics: ####
            N_m[i] += N_m[i-1] - act_N
            Na_m[i] += Na_m[i-1] + div_Na + act_N - unbound_Na*(1 - Na_div_flag)*Na_m[i-1]
            
            if k == 1:
                M_m[i] += M_m[i-1] - div_M
                act_M += -div_M
                
            Ma_m[i] += Ma_m[i-1] + div_Ma + (unbound_Na*(1 - Na_div_flag)*(diff_NaM) if k == 0 else 0) + (diff_MM if k == 1 else 0) + diff_EM
            E_m[i] += E_m[i-1] + div_E + (unbound_Na*(1 - Na_div_flag)*(Na_m[i-1] - diff_NaM) if k == 0 else (2*div_M - diff_MM)) - (die_E + diff_EM)
            
            # Update division flag to allow division to proceed
            Na_div_flag = 1*(Na_m[i] < max_Na)*(b_Na_div > 0)*(Na_m[i] > 0)
            div_E_count += (div_Na if k == 0 else 0) + (div_M if k == 1 else 0) + div_E + div_Ma
            diff_EpM_count += diff_EM
            
            #### New binding events ####
            b_N_act = f_XtoY(sig_1 = p_tcr*I[i-1], sig_2 = p_cyt*H[i-1], psi_1 = 0.0, psi_2 = 4.0, psi_1_2 = 0.0, F_0 = -0.0, K_1 = K_EI, K_2 = K_EH)*Ain[i]/(A_init*char_times[0])*(Ain[i] >= 1)
            b_unbind_t = np.fmin(2*(Na_m[i] == 1)*(i - np.argmin(N_m[0:i], axis = 0))*dt/(f_XtoY(sig_1 = p_tcr*I[i-1], sig_2 = p_cyt*H[i-1], psi_1 = psi_Nact_I, psi_2 = psi_Nact_H, psi_1_2 = psi_Nact_IH, F_0 = 0.0, K_1 = K_IE, K_2 = K_EH)*char_times[1]*2/np.sqrt(np.pi))**2, 1/dt) if np.sum(Na_m[i]) >= 1 else 0.0
            unbound_Na += np.random.binomial(1-unbound_Na, b_unbind_t*dt)
            
            if k == 1:
                b_M_act = f_XtoY(sig_1 = p_tcr*I[i-1], sig_2 = p_cyt*H[i-1], psi_1 = 0.0, psi_2 = 4.0, psi_1_2 = 0.0, F_0 = -0.0, K_1 = K_EI, K_2 = K_EH)*Ain[i]/(A_init*char_times[0])*(Ain[i] >= 1)
            
            #### MYC Dynamics ####
            mycNa = (mycNa + dt*(b_myc*(1-unbound_Na)))*(Na_m[i] > 0)
            # (1-H[i]**l_H/((K_EH/p_cyt)**l_H + H[i]**l_H))
            
            mycE = (mycE - dt*(f_XtoY(sig_1 = p_tcr*I[i], sig_2 = p_cyt*H[i], psi_1 = -psi_Ediv_I, psi_2 = -psi_Ediv_H, psi_1_2 = -psi_Ediv_IH, F_0 = -F0_Ediv, K_1 = K_EI, K_2 = K_EH)*mycE*d_myc))*(E_m[i] >= 1) + (mycNa*(Na_m[i] > 0) if k == 0 else mycM*(M_m[i] > 0))

            mycMa = (mycMa - dt*f_XtoY(sig_1 = p_tcr*I[i], sig_2 = p_cyt*H[i], psi_1 = -psi_Ediv_I, psi_2 = -psi_Ediv_H, psi_1_2 = -psi_Ediv_IH, F_0 = -F0_Ediv, K_1 = K_EI, K_2 = 1*K_EH)*(mycMa*d_myc))*(Ma_m[i] > 0) + (mycNa*(Na_m[i] >= 1) if k == 0 else mycM*(M_m[i] > 0)) # higher decay rate of myc
            
            if k == 1:
                mycM = (mycM + 0*dt*(b_myc*Ain[i]/(M_pop + Ain[i])))*(M_m[i] >= 1)

            #### Time-dependent rates modulated by antigen and cytokine signals ####
            b_Na_div = (mycNa >= myc_thresh)/char_times[2]
            b_E_div = (mycE >= myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
            b_Ma_div = (mycMa >= myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]

            #### Transition probabilities modulated by antigen and cytokine signals ####
            p_Nact[i] += 1 - b_unbind_t*dt
            p_NaM[i] += (p_NaM[i-1] + 2*(dt**2)*f_XtoY(sig_1 = p_tcr*(1-unbound_Na), sig_2 = p_cyt*H[i], psi_1 = psi_NM_I, psi_2 = psi_NM_H, psi_1_2 = psi_NM_IH, F_0 = F0_NM, K_1 = 1/2, K_2 = K_EH)/(2/np.sqrt(np.pi)*char_times[5]/rel_NE_to_M_diff)**2)*(1-unbound_Na)*(Na_m[i] == 1)
            p_EM[i] += p_EM[i-1] + 2*(dt**2)*f_XtoY(sig_1 = p_tcr*(I[i] + S[i]*(d_I == 0.0)), sig_2 = p_cyt*H[i], psi_1 = psi_EM_I, psi_2 = psi_EM_H, psi_1_2 = psi_EM_IH, F_0 = F0_EM, K_1 = K_EI + np.sum(E_m[i]), K_2 = K_EH)*(E_m[i] > 0)/(2/np.sqrt(np.pi)*char_times[5]**2)
            p_Ediv[i] += b_E_div*dt

            if k == 1:
                p_MM[i] += (p_MM[i-1] + 2*(dt**2)*f_XtoY(sig_1 = p_tcr*(I[i] + S[i]*(d_I == 0.0)), sig_2 = p_cyt*H[i], psi_1 = psi_NM_I, psi_2 = psi_NM_H, psi_1_2 = psi_NM_IH, F_0 = F0_NM, K_1 = K_EI + np.sum(M_m[i]), K_2 = K_EH)/(2/np.sqrt(np.pi)*char_times[5]/rel_NE_to_M_diff)**2)*(M_m[i] > 0)
            
            # store time cells become effector
            if k == 0:
                T_E += dt*(E_m[i] > 0) if np.sum(E_m[i]) > 0 else 0.0
                d_E_die = 2*(E_m[i] > 0)*T_E/(2/np.sqrt(np.pi)*char_times[6])**2

                # store time an effector spends in cytotoxic state
                for l in np.arange(0, N_0_var):
                    T_Ecyt[l] += diff_NaM[l]*[0.0] + diff_EM[l]*[T_E[l].item()] + div_Ma[l]*[0.0]
                        
            elif k == 1:
                T_E += dt*(E_m[i] > 0) if np.sum(E_m[i]) > 0 else 0.0
                d_E_die = 2*(E_m[i] > 0)*(T_E)/char_times[6]**2

            #### Store myc levels ####
            if out_data == "full":
                mycNa_m[i] += mycNa
                mycMa_m[i] += mycMa
                mycE_m[i] += mycE

                if k == 1:
                    mycM_m[i] = mycM

            #### Store differentiation biases
            bias_t[i] += np.array([np.mean((p_Nact[i])),
                                   np.mean((p_NaM[i])[Na_m[i]*(1-unbound_Na) > 0]) if np.sum(Na_m[i]*(1-unbound_Na)) > 0.0 else 0.0, 
                                   np.mean((p_EM[i])[E_m[i] > 0]) if np.sum(E_m[i]) > 0.0 else 0.0, 
                                   np.mean((p_MM[i])[M_m[i] > 0]) if k == 1 else 0.0,
                                   np.mean(p_Ediv[i][E_m[i] > 0]) if np.sum(E_m[i]) > 0.0 else 0.0])
            
        # Increment time
            t += dt
        
        # Collect population dynamics
        N, Na, Ma, E = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(Ma_m, axis = 1), np.sum(E_m, axis = 1)
        
        if k == 1:
            M = np.sum(M_m, axis = 1)
        
        if k == 0 and N_0_var > 0:
            T_pEcytM = np.concatenate(T_Ecyt) # time that memory spends in effector
            # Project out surviving primary memory
            M_survive = np.random.binomial( np.ones(len(T_pEcytM), dtype =np.int32), np.exp(- 0.5*(1 + (rel_persist_M - 1)*T_pEcytM/char_times[6])) ) # second infection happens 0.5/d_cM years on average.
            pM_count = int(np.sum(M_survive))
    
        lineage_comp = np.vstack([np.amax(N_m if k == 0 else (M_m + Ma_m), axis = 0) ,
                                  np.amax(Ma_m if k == 0 else (M_m + Ma_m), axis = 0),
                                  np.amax(E_m, axis = 0),
                                  p_tcr*np.ones(N_0_var if k == 0 else pM_count),
                                  p_cyt*np.ones(N_0_var if k == 0 else pM_count)])
        
        if k == 0: # primary infection
            dyn_data = np.array([S, I, Ain, Na, E, Ma, H, I_d_I + I_d_IE, I_d_S]).T
            prim_bias = bias_t #[p_NaM, p_EM]
        elif k == 1: # secondary infection
            dyn_data = np.hstack((dyn_data, np.array([S, I, Ain, Na + M, E, Ma, H, I_d_I + I_d_IE, I_d_S]).T ))
            sec_bias = bias_t # [p_NaM, p_EM, p_MM]
                                 
    ts = np.linspace(0, duration, int(steps) + 1)
    
    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, sS, pI, sI, Ain, N, pE, sE, pM, sM, pH, sH, pI_d_I, sI_d_I, pI_d_S, sI_d_S = dyn_data[:,0], dyn_data[:,-9], dyn_data[:,1], dyn_data[:,-8], dyn_data[:,2], dyn_data[:, 3], dyn_data[:,4], dyn_data[:,-5], dyn_data[:,5], dyn_data[:,-4], dyn_data[:,6], dyn_data[:,-3], dyn_data[:,7], dyn_data[:,-2], dyn_data[:,8], dyn_data[:,-1]
        
    dt = ts[1]-ts[0]
    
    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, d_IH, K_IE, K_IH,
              A_init, b_Ain, b_H, d_H, K_EI, K_EH,
              N_0, max_Na, b_myc, d_myc, myc_thresh]),
              char_times,
              activation_regulation, NM_regulation, EM_regulation, expansion_regulation))

    sim_summary = np.array([np.sum(pI*dt), 
                       np.sum(sI*dt),
                       np.argmax(pI)*dt,
                       np.argmax(pI < I_0)*dt,
                       np.amax(pI_d_I), 
                       np.amax(sI_d_I), 
                       np.amax(pI_d_S),
                       np.amax(sI_d_S),
                       np.max(pE),
                       np.max(sE),
                       np.argmax(pE)*dt,
                       np.argmax(sE)*dt,
                       pM[-1],
                       sM[-1],
                       pM_count if N_0 > 0 else 0.0,
                       np.sum(pE*dt), 
                       np.sum(sE*dt),
                       np.sum(pH*dt),
                       np.sum(sH*dt),
                       np.amin(pS),
                       np.amin(sS)])

    # down-size timeseries
    pnts = int(0.1*steps)
    keep = [i*10 for i in np.arange(0,pnts)]
    
    if out_data == "full":
        out_dict = {"reg_coeffs": np.concatenate((activation_regulation, NM_regulation, EM_regulation, expansion_regulation)), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "sec_diff_bias": sec_bias if k == 1 else [],"eff_by_lin": (E_m), "Na_myc_by_lin": mycM_m if k == 1 else mycNa_m, "Ma_myc_by_lin": mycMa_m, "E_myc_by_lin": mycE_m, "parameters": parameters, "summary_stats": sim_summary, "pmemory_formed": T_pEcytM if N_0 > 0 else []}
    elif out_data == "small":
        out_dict = {"cell_time_series": dyn_data[keep], "prim_diff_bias": prim_bias[keep], "sec_diff_bias": sec_bias[keep] if k == 1 else [], "parameters": parameters, "summary_stats": sim_summary}
    
    return out_dict


stat_names = [r"$\int_0^{T_{sim}} I_{p}dt$",
              r"$\int_0^{T_{sim}} I_{s}dt$",
              r"$T_{I_p}^{max}$",
              r"$T_{I_p}^{min}$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{p}dt$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{s}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{p}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{s}dt$",
              r"$E_p^{max}$",
              r"$E_s^{max}$",
              r"$T_{E_p}^{max}$",
              r"$T_{E_s}^{max}$",
              r"$(M_p)^\infty$",
              r"$(M_s)^\infty$",
              r"$M(0)$",
              r"$\int E_p dt$",
              r"$\int E_s dt$",
              r"$\int H_p dt$",
              r"$\int H_s dt$",
              r"$S_p^{min}$",
              r"$S_s^{min}$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$A_{out}^{(0)}$", r"$b_{A_in}$", r"$b_H$", r"$d_H$", r"$K_{E,I}$", r"$K_{E,H}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N^*,A_{in}}^{(+)}$", r"$\tau_{N^*,A_{in}}^{(-)}$", r"$\tau_{N^*}$", r"$\tau_{E_{div}}$", r"$\tau_{M_{div}}$", r"$\tau_{M_{diff}}$", r"$\tau_{E_{die}}$", r"$\tau_{E_{cyt}}$",
               r"$\psi_{N^*}^{(I)}$", r"$\psi_{N^*}^{(H)}$", r"$\psi_{N^*}^{(I,H)}$",
               r"$\psi_{N,M}^{(I)}$", r"$\psi_{N,M}^{(H)}$", r"$\psi_{N,M}^{(I,H)}$", 
               r"$\psi_{E,M}^{(I)}$", r"$\psi_{E,M}^{(H)}$", r"$\psi_{E,M}^{(I,H)}$",
               r"$\psi_{E^*}^{(I)}$", r"$\psi_{E^*}^{(H)}$", r"$\psi_{E^*}^{(I,H)}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'd_IH', 'K_IE',
                      'K_IH', 'A_init', 'b_Ain', 'b_H', 'd_H', 'K_EI', 'K_EH',
                      'N_0', 'max_Na', 'b_myc', 'd_myc', 'myc_thresh',
                      't_act', 't_bind', 't_Na_div', 't_E_div', 't_M_div', 't_EM_diff', 't_E_die', 't_E_cyt',
                      'psi_Nact_I', 'psi_Nact_H', 'psi_Nact_IH',
                      'psi_NM_I', 'psi_NM_H', 'psi_NM_IH',
                      'psi_EM_I', 'psi_EM_H', 'psi_EM_IH',
                      'psi_Ediv_I', 'psi_Ediv_H', 'psi_Ediv_IH']

stat_names_for_df = ['p_load', 's_load','T_max_pI', 'T_min_pI', 'harm_pI', 'harm_sI', 
                     'harm_pS', 'harm_sS', 'max_pE', 'max_sE','T_max_pE', 'T_max_sE', 
                     'inf_pM', 'inf_sM', 'init_M','int_pE', 'int_sE',
                     'int_pH', 'int_sH', 'min_pS', 'min_sS']