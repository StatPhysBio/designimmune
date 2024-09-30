import itertools

### Stochastic model dynamics ###
import numpy as np
from scipy import stats
import os as os
import pandas as pd
import itertools
from scipy.optimize import fsolve

### (1) Define simulation parameters
# Define simulation parameters
sim_duration = 21
sim_steps = int(0.20*(10**4))

# infection dynamics
S_0 = 10**7 #susceptible cells
d_S = 0.01
b_I = (10**(-7)) # harm per unit virion (Chao et al. 2004, Iwami et al. 2015)
I_0 = 1000 # initial detectable levelof infected cells
d_IE = 16 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE = 10**4 # effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
K_EI = K_IE
d_I = np.minimum(25*d_S, S_0*b_I) # successful virus cannot kill cells faster than it infects new ones

# Inflammatory response
d_H = 2.0
b_H = 1 # innate response/inflammation per lysed cell compared to natural death
K_IH = d_S*S_0/d_H # half-max level of instantaneous damage required to trigger innate/inflammatory response
K_EH = 1*K_IH # half-max level of inflammation required to trigger lymphocyte response
K_SE = 10*S_0
kappa = 0.0 # maximal reduction in replication rate due to inflammatory response
d_IH = d_IE*0

# Immune cells
N_0 = 200
max_Na = 2**2
max_expand = 2**15 - 1 #(Marchingo et al.)
t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_act = 0.5, 1.0, 1/4, 1/3, 1/2, 2.5, 1/4
rel_persist_M = 5
t_long_M = 5 # years
t_short_M = 0.8 # years

# Division timer
d_myc = 1/t_E_die
myc_thresh = 1.0
b_myc = myc_thresh/t_act # Prlic et al. (2006)

# hyper parameters
alpha = 0.5 # weight of antigenic signals relative to inflamatory signals
num_pnts = 7
vir_prop = np.array(np.meshgrid(d_S*np.linspace(10, 100, num_pnts), # vary d_I
                                   K_IE*np.logspace(0.0, 2, num_pnts), # vary K_IE
                                   b_I*np.linspace(0.75, 2.0, num_pnts) # vary b_I
                                           )).T.reshape(-1,3)

vir_prop_select = vir_prop[vir_prop[:,2]*S_0 > vir_prop[:,0]]

# define reg options
psi_max = 4.0
psi_2d_full = np.array(list(itertools.product(np.linspace(-psi_max/2, psi_max/2, int(psi_max + 1)).tolist(),
                                 np.linspace(-psi_max/2, psi_max/2, int(psi_max + 1)).tolist(),
                                 np.linspace(-psi_max/2, psi_max/2, int(psi_max + 1)).tolist(),
                                 np.linspace(-psi_max/2, psi_max/2, int(psi_max + 1)).tolist())))

psi_2d = psi_2d_full[(np.abs(psi_2d_full[:,0]) + np.abs(psi_2d_full[:,1]) + 2*np.abs(psi_2d_full[:,2]) <= psi_max)]
psi_2d_pos = psi_2d[(psi_2d[:,0] >= 0)*(psi_2d[:,1] >= 0)*(psi_2d[:,2] >= 0)]
# psi_2d_full_pos = np.array(list(itertools.product(np.linspace(0, psi_max/2, int(psi_max + 1)).tolist(),
#                                  np.linspace(0, psi_max/2, int(psi_max + 1)).tolist(),
#                                  np.linspace(0, psi_max/2, int(psi_max + 1)).tolist(),
#                                  np.linspace(psi_max/2, psi_max/2, int(psi_max + 1)).tolist())))
# psi_2d_pos = psi_2d_full_pos[(np.abs(psi_2d_full_pos[:,0]) + np.abs(psi_2d_full_pos[:,1]) + 2*np.abs(psi_2d_full_pos[:,2]) <= psi_max)]

psi_2d_comp = psi_2d[(psi_2d[:,2] == 0)]
psi_2d_nobias = psi_2d[(psi_2d[:,3] == 0)]
psi_2d_comp_bias = np.array(list(itertools.product([0.0],[ 0.0], [0.0], np.linspace(-(1 + psi_max), (1 + psi_max), 11).tolist())))

bl_block = list(itertools.product([0.0],[ 0.0], [0.0], np.linspace(-psi_max, psi_max, int(psi_max + 1)).tolist()))
psi_2d_sparse = np.vstack((np.array(list(itertools.product(psi_2d.tolist(), bl_block, bl_block, bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, psi_2d.tolist(), bl_block, bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, bl_block, psi_2d.tolist(), bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, bl_block, bl_block, psi_2d.tolist()))).reshape(-1,16)))

NM_psis = [0.0, 0.0, 0.0, 0.0] # regulatory weights: psi_M_I, psi_M_H, psi_M_IH
EM_psis = [0.0, 0.0, 0.0, 0.0]
act_psis = [psi_max/2, psi_max/2, 0.0, 0.0]
exp_psis = [psi_max/2, psi_max/2, 0.0, 0.0]

### (2) Define functions for simulations
# Define functions for simulations
def f_XtoY(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, psi_1_2 = 0.0, psi_1_3 = 0.0, psi_2_3 = 0.0, F_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0, reg_model = "mwc_like"):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    # reg_model := family of regulatory functions considered: Monod-Wyman-Changeaux inspired, and Hill functions
    
    if reg_model == "mwc_like":
        F_1 = psi_1*np.log((1 + sig_1/K_1)) + psi_2*np.log((1 + sig_2/K_2)) + psi_3*np.log((1 + sig_3/K_3)) + psi_1_2*np.log((1 + (sig_1*sig_2)/(K_1*K_2))) + psi_1_3*np.log((1 + (sig_1*sig_3)/(K_1*K_3))) + psi_2_3*np.log((1 + (sig_2*sig_3)/(K_2*K_3)))
        out = 1/(1 + np.exp(- (F_1 + F_0)))

    elif reg_model == "competition_model":
        F_1 = (psi_1*np.log((1 + sig_1/K_1))**2 + psi_2*np.log((1 + sig_2/K_2))**2 + psi_3*np.log((1 + sig_3/K_3))**2)/(2*(np.log((1 + sig_1/K_1)) + np.log((1 + sig_2/K_2)) + np.log((1 + sig_3/K_3)))) + (psi_1*np.log((1 + sig_1/K_1)) + psi_2*np.log((1 + sig_2/K_2)) + psi_3*np.log((1 + sig_3/K_3)))/2
        out = 1/(1 + np.exp(- (F_1 + F_0)))
        
    else:
        out = 0.0
    
    return out

def memory_duration(t, T_eff, T_degrade = t_E_die, target = N_0):

    out = np.exp(- t/np.maximum(t_long_M - (t_long_M - t_short_M)*T_eff/T_degrade, sim_duration/365.25))

    return np.sum(out) - target

def antigenicity_over_harm(df):
    out = np.log(0 + (df['K_EH'] if 'K_EH' in df.columns.tolist() else K_EH)*(df['b_I']*(df['S_0'] if 'S_0' in df.columns.tolist() else S_0) - df['d_I'] + (df['d_H'] if 'd_H' in df.columns.tolist() else d_H) )/(df['d_I']*(df['I_0'] if 'I_0' in df.columns.tolist() else I_0)))/np.log(df['K_IE']/(df['I_0'] if 'I_0' in df.columns.tolist() else I_0))
    return out
    
#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################

def lin_stoch_sim(S_0 = S_0, I_0 = I_0, b_I = b_I, d_S = d_S, d_I = d_I, d_IE = d_IE, d_IH = d_IH, K_IE = K_IE, K_IH = K_IH, K_SE = K_SE,
                    b_H = b_H, d_H = d_H, K_EI = K_EI, K_EH = K_EH, kappa = kappa,
                    N_0 = N_0, max_Na = max_Na, b_myc = b_myc, d_myc = d_myc, myc_thresh = myc_thresh, max_expand = max_expand,
                    char_times = [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_act],
                    NM_regulation = NM_psis,
                    EM_regulation = EM_psis,
                    activation_regulation = act_psis,
                    expansion_regulation = exp_psis,
                    alpha = alpha,
                    infection = "prim",
                    vir_model = "indep_harm",
                    duration = sim_duration, 
                    steps = sim_steps,
                    reg_model = "mwc_like",
                    out_data = "small"):
    
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, d_IH := d_IH, K_IE := K_IE, K_IH := K_IH,
    # b_H := b_H, d_H := d_H, K_EI := K_EI, K_EH := K_EH,
    # N_0 := N_0, max_Na := max_Na, b_myc := b_myc, d_myc := d_myc, myc_thresh := myc_thresh,
    # char_times := [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die, t_Nact],
    
    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_0*d_S
    
    # draw population of reponding cells for agent-based simulations
    psi_Nact_I, psi_Nact_H, psi_Nact_IH, F0_Nact = activation_regulation[0], activation_regulation[1], activation_regulation[2], activation_regulation[3]
    psi_NM_I, psi_NM_H, psi_NM_IH, F0_NM = NM_regulation[0], NM_regulation[1], NM_regulation[2], NM_regulation[3]
    psi_EM_I, psi_EM_H, psi_EM_IH, F0_EM = EM_regulation[0], EM_regulation[1], EM_regulation[2], EM_regulation[3]
    psi_Ediv_I, psi_Ediv_H, psi_Ediv_IH, F0_Ediv = expansion_regulation[0], expansion_regulation[1], expansion_regulation[2], expansion_regulation[3]
    
    p_tcr = 1.0 #np.ones(N_0_var)
    p_cyt = 1.0 #np.ones(N_0_var)
        
    # define variables for storage
    I = np.zeros(int(steps)+1)
    S = np.zeros(int(steps)+1)
    V = np.zeros(int(steps)+1)
    H = np.zeros(int(steps)+1)
    I_d_I = np.zeros(int(steps)+1)
    I_d_IE = np.zeros(int(steps)+1)
    I_d_S = np.zeros(int(steps)+1)
    
    N_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32)
    N_m[0,:] +=1
    T_Ecyt = [[] for i in np.arange(0, N_0_var)]

    div_E_count = np.zeros(N_0_var)
    diff_EpM_count = np.zeros(N_0_var, dtype = np.int32)
    
    Na_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32)
    Ma_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32)
    E_m = np.zeros((int(steps)+1, N_0_var), dtype = np.int32) # effector in periphary
    T_E = np.zeros(N_0_var)
    
    if out_data == "full":
        mycN_m = np.zeros((int(steps)+1, N_0_var))
        mycMa_m = np.zeros((int(steps)+1, N_0_var))
        mycE_m = np.zeros((int(steps)+1, N_0_var))
    
    bias_t = np.zeros((int(steps)+1, 4))
    
    # Define event timer variables
    bound_N = np.zeros(N_0_var, dtype =np.int32)
    Na_div_flag = np.ones(N_0_var, dtype =np.int32)
    Na_flag = np.ones(N_0_var, dtype =np.int32)
    
    mycN = np.zeros(N_0_var)
    mycE = np.zeros(N_0_var)
    mycMa = np.zeros(N_0_var)
    
    r_NaM = np.zeros((int(steps)+1, N_0_var))
    r_EM = np.zeros((int(steps)+1, N_0_var))
    r_Nact = np.zeros((int(steps)+1, N_0_var))
    r_Ediv = np.zeros((int(steps)+1, N_0_var))
    
    b_unbind_t = np.zeros(N_0_var)
    b_N_bind = np.zeros(N_0_var)
    b_Na_div = np.ones(N_0_var)/char_times[2]
    b_myc_t = np.zeros(N_0_var)
    b_E_div = np.ones(N_0_var)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
    b_Ma_div = np.ones(N_0_var)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]
    d_E_die = np.ones(N_0_var)/char_times[5]   
    
    #################################
    ### RUN POPULATION SIMULATION ###
    #################################
    t = 0.0
    S[0] = S_0
    I[0] = I_0 if vir_model != 'autoimmune' else 0.0
    V[0] = 0
    H[0] = 0.0
    
    for i in np.arange(1, int(steps) + 1):
        # select virulence model:
        if vir_model == "dep_harm": # makes  average virus produced roughly the same independent of infected death rate
            b_I_t = b_I + d_I/S_0
            d_Sauto = 0
            t_Hauto = 1.0
        elif vir_model == "autoimmune":
            b_I_t = 0.0
            d_Sauto = d_I 
            K_SE = K_IE
            I_0 = 0.0
            t_Hauto = 1.0 # duration of autoimmune inflammation
        else:
            b_I_t = b_I
            d_Sauto = 0
            t_Hauto = 1.0
    
        # Compute total population of cell types
        E_pop = np.sum(E_m[i-1])
        
            
        #### Run infection dynamics: replication and effector clearance ####
        
        # Update state of susceptible, infected and inflammation
        # (a) event variables
        # S_to_I, I_die = np.random.binomial(S[i-1], (I[i-1] >= I_0)*b_I_t*I[i-1]*dt), np.random.binomial(I[i-1], dt*(d_IE*(E_pop)/(K_IE + I[i-1] + E_pop) + d_I))
        # I_d_IE[i] += I_d_IE[i-1] + np.random.binomial(I_die, d_IE*(E_pop)/(K_IE + I[i-1] + E_pop)/( d_IE*(E_pop)/(K_IE + I[i-1] + E_pop) + d_I)) # cells killed by immune response
        # I_d_I[i] += I_d_I[i-1] + (I_die - (I_d_IE[i] - I_d_IE[i-1])) + (I[i] if i == int(steps) else 0) # cells killed by infection
            
        S_to_I, I_die = dt*S[i-1]*(I[i-1] >= I_0)*b_I_t*I[i-1], dt*I[i-1]*(d_IE*(E_pop)/(K_IE + I[i-1] + E_pop) + d_I)
        I_d_IE[i] += I_d_IE[i-1] + dt*I[i-1]*d_IE*(E_pop)/(K_IE + I[i-1] + E_pop) # cells killed by immune response
        I_d_I[i] += I_d_I[i-1] + dt*I[i-1]*d_I + (I[i] if i == int(steps) else 0) # cells killed by infection
        I_d_S[i] += I_d_S[i-1] + dt*S[i-1]*( d_IE*(E_pop)/(K_SE + S[i-1] + E_pop) )
        
        # (b) state variables
        S[i] += S[i-1] - S_to_I + dt*(b_S - (d_S + d_Sauto*np.exp(-(t/t_Hauto)**2))*S[i-1] - S[i-1]*d_IE*(E_pop)/(K_SE + S[i-1] + E_pop))*(S[i-1] >= 1.0)
        
        I[i] += I[i-1] + S_to_I - I_die

        harm_detected = I_die + dt*S[i-1]*d_Sauto*np.exp(-(t/t_Hauto)**2)

        cells_sensed = (I[i] + S[i]*(vir_model == "autoimmune"))
        
        H[i] += H[i-1] + b_H*(harm_detected) - dt*d_H*H[i-1]*(H[i-1] >= 0.0)
        
        ## I. Recruitment/Priming

        # (a) Phase 1: Naive cells encounter and bind APCs
        if cells_sensed >= 1:
            bound_N += np.random.binomial(N_m[i-1], b_N_bind*dt) - np.random.binomial(N_m[i-1], b_unbind_t*dt)
        else:
            bound_N = 0*bound_N

        # (b) Phase 2: Activated naive cells are bound to APCs and receive stimulation.
        act_N = (1 - bound_N)*(mycN >= myc_thresh)*(N_m[i-1] > 0)

        # (c) Phase 3: Unbound activated naive cells divide and then differentiate
        div_Na = np.random.binomial(Na_m[i-1]*Na_div_flag, dt*b_Na_div)

        diff_NaM = np.random.binomial(Na_m[i-1]*(1 - Na_div_flag), 1 - np.exp(-np.sum(r_NaM[0:i-1], axis = 0)*dt) if np.sum(Na_m[i-1]) > 0 else 0)
        
        ## II. Expansion

        # (a) New central memory cells divide
        div_Ma = np.random.binomial(Ma_m[i-1] - diff_EpM_count, dt*b_Ma_div)

        # (b) Effector cells divide, differentiate, die
        die_E = np.random.binomial(E_m[i-1], d_E_die*dt)
        div_E = np.random.binomial(E_m[i-1] - die_E, dt*(b_E_div))
        diff_EM = np.random.binomial(E_m[i-1] - die_E, dt*r_EM[i-1])
        
        #### Update population dynamics: ####
        N_m[i] += N_m[i-1] - act_N
        Na_m[i] += Na_m[i-1] + div_Na + act_N - (1 - Na_div_flag)*Na_m[i-1]
            
        Ma_m[i] += Ma_m[i-1] + div_Ma + (1 - Na_div_flag)*(diff_NaM) + diff_EM
        E_m[i] += E_m[i-1] + div_E + (1 - Na_div_flag)*(Na_m[i-1] - diff_NaM) - (die_E + diff_EM)
        
        # Update division flag to allow division to proceed
        Na_div_flag = (Na_m[i] < max_Na)*(Na_m[i] > 0)
        div_E_count += div_Na + div_E + div_Ma
        diff_EpM_count += diff_EM
        
        #### New binding events ####
        
        b_N_bind = (N_m[i] > 0)*(1 - bound_N)*f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = psi_max/4, psi_2 = psi_max/2, psi_1_2 = 0.0, F_0 = -2.0, K_1 = S_0, K_2 = K_EH, reg_model = reg_model)/char_times[0] if cells_sensed >= 1 else 0.0

        b_unbind_t = bound_N*np.fmin(2*(i - np.argmin(N_m[0:i], axis = 0))*dt/(f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = psi_max/2, psi_2 = psi_max/4, psi_1_2 = 0.0, F_0 = -2.0, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)*char_times[1]*2/np.sqrt(np.pi))**2, 1/dt) if np.sum(bound_N) >= 1 else 0.0
        
        b_myc_t = b_myc*f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = psi_Nact_I, psi_2 = psi_Nact_H, psi_1_2 = psi_Nact_IH, F_0 = F0_Nact, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)*bound_N # Maybe: multiply by np.sqrt(np.pi)/2. Can see argument for leaving as a linear rate
        
        #### MYC Dynamics ####
        mycN = (mycN + dt*(b_myc_t - (1 - bound_N)*mycN*d_myc*(mycN > 0)))*(N_m[i] + Na_m[i] > 0)
        
        mycE = (mycE - dt*(f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = -psi_Ediv_I, psi_2 = -psi_Ediv_H, psi_1_2 = -psi_Ediv_IH, F_0 = -F0_Ediv, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)*mycE*d_myc))*(E_m[i] >= 1) + mycN*(Na_m[i] > 0)

        mycMa = (mycMa - dt*f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = -psi_Ediv_I, psi_2 = -psi_Ediv_H, psi_1_2 = -psi_Ediv_IH, F_0 = -F0_Ediv, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)*(mycMa*d_myc))*(Ma_m[i] > 0) + mycN*(Na_m[i] >= 1) # higher decay rate of myc

        #### Time-dependent rates modulated by antigen and cytokine signals ####
        b_Na_div = 1/char_times[2]
        b_E_div = (mycE >= myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
        b_Ma_div = (mycMa >= myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]
        
        #### Transition probabilities modulated by antigen and cytokine signals ####
        r_Nact[i] += b_myc_t/myc_thresh
        
        r_NaM[i] += (r_NaM[i-1] + 2*dt*f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = psi_NM_I, psi_2 = psi_NM_H, psi_1_2 = psi_NM_IH, F_0 = F0_NM, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)/(2/np.sqrt(np.pi)*(char_times[1] - char_times[6]))**2)*(N_m[i]*bound_N > 0)*(mycN >= myc_thresh)
        
        r_EM[i] += (r_EM[i-1] + 2*dt*f_XtoY(sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], psi_1 = psi_EM_I, psi_2 = psi_EM_H, psi_1_2 = psi_EM_IH, F_0 = F0_EM, K_1 = (K_EI*(vir_model != "autoimmune") + K_SE*(vir_model == "autoimmune")), K_2 = K_EH, reg_model = reg_model)/(2/np.sqrt(np.pi)*char_times[5]**2))*(E_m[i] > 0) # Urgent: need to square entire denominator
        
        r_Ediv[i] += b_E_div
        
        # store time cells become effector
        T_E += dt*(E_m[i] > 0) if np.sum(E_m[i]) > 0 else 0.0
        d_E_die = 2*(E_m[i] > 0)*T_E/(2/np.sqrt(np.pi)*char_times[5])**2

        # store time an effector spends in cytotoxic state
        for l in np.where(N_m[i] == 0)[0]:
            T_Ecyt[l] += diff_NaM[l]*[0.0] + diff_EM[l]*[T_E[l].item()] + div_Ma[l]*[0.0]

        #### Store myc levels ####
        if out_data == "full":
            mycN_m[i] += mycN
            mycMa_m[i] += mycMa
            mycE_m[i] += mycE

        #### Store differentiation biases
        bias_t[i] += np.array([np.mean((r_Nact[i])),
                               np.mean((r_NaM[i])[N_m[i]*bound_N > 0]) if np.sum(N_m[i]*bound_N) > 0.0 else 0.0, 
                               np.mean((r_EM[i])[E_m[i] > 0]) if np.sum(E_m[i]) > 0.0 else 0.0, 
                               np.mean(r_Ediv[i][E_m[i] > 0]) if np.sum(E_m[i]) > 0.0 else 0.0])
        
        # Increment time
        t += dt
        
    # Collect population dynamics
    N, Na, Ma, E = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(Ma_m, axis = 1), np.sum(E_m, axis = 1)
    
    T_pEcytM = np.concatenate(T_Ecyt) if N_0 > 0 else np.array([0.0]) # time that memory spends in effector
    # Project out surviving primary memory
    M_duration = fsolve(memory_duration, 1.0, args = (T_pEcytM, char_times[5], N_0), xtol = 0.001).item() if len(T_pEcytM) > N_0 and N_0 > 0.0 else 0.0

    lineage_comp = np.vstack([np.amax(N_m, axis = 0) ,
                              np.amax(Ma_m, axis = 0),
                              np.amax(E_m, axis = 0),
                              p_tcr*np.ones(N_0_var),
                              p_cyt*np.ones(N_0_var)])
    
    dyn_data = np.array([S, I, N, Na, E, Ma, H, I_d_I + I_d_IE, I_d_S]).T
    prim_bias = bias_t #[r_NaM, r_EM]
                                 
    ts = np.linspace(0, duration, int(steps) + 1)
    
    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, pI, N, Na, pE, pM, pH, pI_d_I, pI_d_S = dyn_data[:,0], dyn_data[:,1], dyn_data[:,2], dyn_data[:, 3], dyn_data[:,4], dyn_data[:,5], dyn_data[:,6], dyn_data[:,7], dyn_data[:,8]
        
    dt = ts[1]-ts[0]
    
    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, d_IH, K_IE, K_IH,
              b_H, d_H, K_EI, K_EH,
              N_0, max_Na, b_myc, d_myc, myc_thresh]),
              char_times,
              activation_regulation, NM_regulation, EM_regulation, expansion_regulation))

    sim_summary = np.array([np.sum(pI*dt), 
                       np.argmax(pI)*dt,
                       np.argmax(pI < I_0)*dt,
                       np.amax(pI_d_I), 
                       np.amax(pI_d_S),
                       np.max(pE),
                       np.argmax(pE)*dt,
                       np.mean(np.argmax(E_m > 0 , axis = 0)*dt*(np.amax(E_m, axis = 0) > 0 )) + sim_duration*(np.max(pE) < 1.0),
                       pM[-1],
                       M_duration if N_0 > 0 else 0.0,
                       np.sum(pE*dt), 
                       np.sum(pH*dt),
                       np.amin(pS)])

    # down-size timeseries
    pnts = int(0.1*steps)
    keep = [i*10 for i in np.arange(0,pnts)]
    
    if out_data == "full":
        out_dict = {"reg_coeffs": np.concatenate((activation_regulation, NM_regulation, EM_regulation, expansion_regulation)), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "eff_by_lin": (E_m), "N_myc_by_lin": mycN_m, "Ma_myc_by_lin": mycMa_m, "E_myc_by_lin": mycE_m, "parameters": parameters, "summary_stats": sim_summary, "pmemory_formed": T_pEcytM if N_0 > 0 else []}
    elif out_data == "small":
        out_dict = {"cell_time_series": dyn_data[keep], "prim_diff_bias": prim_bias[keep],"parameters": parameters, "summary_stats": sim_summary}
    
    return out_dict


stat_names = [r"$\int_0^{T_{sim}} I_{p}dt$",
              r"$T_{I_p}^{max}$",
              r"$T_{I_p}^{min}$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{p}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{p}dt$",
              r"$E_p^{max}$",
              r"$T_{E_p}^{max}$",
              r"$T_{E_p}^{start}$",
              r"$(M_p)^{max}$",
              r"$T_{M < N}$",
              r"$\int E_p dt$",
              r"$\int H_p dt$",
              r"$S_p^{min}$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$b_H$", r"$d_H$", r"$K_{E,I}$", r"$K_{E,H}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N,A_{in}}$", r"$\tau_{N \cdot A_{in}}$", r"$\tau_{N^*,N^*}$", r"$\tau_{E_,E}$", r"$\tau_{M,M}$", r"$\tau_{E_{die}}$", r"$\tau_{N^*}$",
               r"$\psi_{N^*}^{(I)}$", r"$\psi_{N^*}^{(H)}$", r"$\psi_{N^*}^{(I,H)}$", r"$F_{N^*}$",
               r"$\psi_{N,M}^{(I)}$", r"$\psi_{N,M}^{(H)}$", r"$\psi_{N,M}^{(I,H)}$", r"$F_{N,M}$", 
               r"$\psi_{E,M}^{(I)}$", r"$\psi_{E,M}^{(H)}$", r"$\psi_{E,M}^{(I,H)}$", r"$F_{E,M}$", 
               r"$\psi_{E^*}^{(I)}$", r"$\psi_{E^*}^{(H)}$", r"$\psi_{E^*}^{(I,H)}$", r"$F_{E^*}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'd_IH', 'K_IE',
                      'K_IH', 'b_H', 'd_H', 'K_EI', 'K_EH',
                      'N_0', 'max_Na', 'b_myc', 'd_myc', 'myc_thresh',
                      't_bind', 't_unbind', 't_Na_div', 't_E_div', 't_M_div', 't_E_die', 't_act',
                      'psi_Nact_I', 'psi_Nact_H', 'psi_Nact_IH', 'F0_Nact',
                      'psi_NM_I', 'psi_NM_H', 'psi_NM_IH', 'F0_NM',
                      'psi_EM_I', 'psi_EM_H', 'psi_EM_IH', 'F0_EM',
                      'psi_Ediv_I', 'psi_Ediv_H', 'psi_Ediv_IH', 'F0_Ediv']

stat_names_for_df = ['p_load', 'T_max_pI', 'T_min_pI', 'harm_pI',
                     'harm_pS', 'max_pE', 'T_pE_max', 'T_pE_start', 
                     'max_pM', 'T_pM_min','int_pE',
                     'int_pH', 'min_pS']

NM_reg = ['psi_NM_I', 'psi_NM_H', 'psi_NM_IH', 'F0_NM']
EM_reg = ['psi_EM_I', 'psi_EM_H', 'psi_EM_IH', 'F0_EM']
Nact_reg = ['psi_Nact_I', 'psi_Nact_H', 'psi_Nact_IH', 'F0_Nact']
Ediv_reg = ['psi_Ediv_I', 'psi_Ediv_H', 'psi_Ediv_IH', 'F0_Ediv']