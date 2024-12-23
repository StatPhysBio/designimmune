import itertools

### Stochastic model dynamics ###
import numpy as np
from scipy import stats
import os as os
import pandas as pd
import itertools
from scipy.optimize import fsolve, minimize

### (1) Define simulation parameters
# Define simulation parameters
sim_duration = 20
sim_steps = int(0.25*(10**4))

# infection dynamics
S_0 = 10**7 #susceptible cells
d_S = 0.01
b_I = (10**(-7)) # harm per unit virion (Chao et al. 2004, Iwami et al. 2015)
I_0 = 1000 # initial detectable levelof infected cells
d_IE = 16 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE = 10**4 # effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
d_I = np.minimum(25*d_S, S_0*b_I) # successful virus cannot kill cells faster than it infects new ones

# Inflammatory response
d_H = 2.0
b_H = 1 # innate response/inflammation per lysed cell compared to natural death
K_EH = d_S*S_0/d_H # half-max level of innate/inflammatory response required to trigger lymphocyte response
K_SE = 10*S_0

# Immune cells
N_0 = 300
max_Na = 2**2
max_expand = 2**14 #(Marchingo et al.)
t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_act, t_bind_eM = 1.0, 3/4, 1/4, 1/3, 1/2, 2.5, 1/4, 1/10
t_long_M = 5*365.25 # years
t_short_M = 0.8*365.25 # years

# Division timer
d_myc = 1.0 # Heinzel et al (2017)
myc_thresh = 1.0
b_myc = myc_thresh/t_act # Prlic et al. (2006)

# Auto-immunity and cancer
t_Hauto = 0.5 # duration of autoimmune inflammation
t_Hcancer = 1.0 # duration of supplied inflammatory cytokines
b_C = 1/10 # growth rate of cancer

# hyper parameters
num_pnts = 10
infection_sample = np.array(np.meshgrid(d_S*np.linspace(25, 100, num_pnts), # vary d_I
                                   K_IE*np.logspace(0.0, 2, num_pnts), # vary K_IE
                                   b_I*np.array([1.5]), # vary b_I
                                   K_EH*np.logspace(0.0, 0.0, 1), # vary K_EH
                                   N_0*np.logspace(0.0, 0.0, 1), # vary N_0
                                   I_0*np.logspace(0.0, 0.0, 1) # vary I_0
                                           )).T.reshape(-1,6)

auto_sample = np.array(np.meshgrid(d_S*np.array([1.0]), # vary d_I
                                   K_SE*np.array([1.0]), # vary K_IE
                                   b_I*np.array([0.0]), # vary b_I
                                   K_EH*np.array([1.0]), # vary K_EH
                                   N_0*np.logspace(-1.0, 1, 5), # vary N_0
                                   I_0*np.array([0.0]) # vary I_0
                                           )).T.reshape(-1,6)

cancer_sample = np.array(np.meshgrid(d_S*np.array([1.0]), # vary d_I
                                   K_SE*np.logspace(-3.0, 0.0, 5), # vary K_IE
                                   b_C*np.array([1.0]), # vary b_I
                                   K_EH*np.array([1.0]), # vary K_EH
                                   N_0*np.logspace(-1.0, 1, 5), # vary N_0
                                   S_0*np.array([0.1]) # vary I_0
                                           )).T.reshape(-1,6)

infection_sample_select = np.vstack([infection_sample[infection_sample[:,2]*S_0 - infection_sample[:,0] >= 1/4],
                                     auto_sample,
                                     cancer_sample])

# define reg options
psi_max = 3.0
F0_max = 4.0
pnts = 5
psi_2d_full = np.array(list(itertools.product(np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-F0_max, F0_max, int(pnts)).tolist())))

psi_2d = psi_2d_full[(np.abs(psi_2d_full[:,0]) + np.abs(psi_2d_full[:,1]) + 2*np.abs(psi_2d_full[:,2]) <= psi_max)]
psi_2d_pos = psi_2d[(psi_2d[:,0] >= 0)*(psi_2d[:,1] >= 0)*(psi_2d[:,2] >= 0)]

psi_2d_comp = psi_2d[(psi_2d[:,2] == 0)]
psi_2d_nobias = psi_2d[(psi_2d[:,3] == 0)]
psi_2d_comp_bias = np.array(list(itertools.product([0.0],[ 0.0], [0.0], np.linspace(-(1 + psi_max), (1 + psi_max), 11).tolist())))

bl_block = list(itertools.product([0.0],[ 0.0], [0.0], np.linspace(-F0_max, F0_max, int(pnts)).tolist()))
psi_2d_sparse = np.vstack((np.array(list(itertools.product(psi_2d_full.tolist(), bl_block, bl_block, bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, psi_2d_full.tolist(), bl_block, bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, bl_block, psi_2d_full.tolist(), bl_block))).reshape(-1,16),
           np.array(list(itertools.product(bl_block, bl_block, bl_block, psi_2d_full.tolist()))).reshape(-1,16)))

act_psis = [psi_max, psi_max, -psi_max, F0_max]
NE_psis = [psi_max, psi_max, -psi_max, F0_max] # regulatory weights: psi_M_I, psi_M_H, psi_M_P
EM_psis = [-psi_max, -psi_max, psi_max, -F0_max]
exp_psis = [psi_max, psi_max, -psi_max, 0.0]

### (2) Define functions for simulations
# Define functions for simulations

def f_XtoY_mwc_like(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, F_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    F_1 = (
        psi_1 * np.log(1 + sig_1 / K_1)
        + psi_2 * np.log(1 + sig_2 / K_2)
        + psi_3 * np.log(1 + sig_3 / K_3)
    )
    out = 1 / (1 + np.exp(-(F_1 + F_0)))
    return out

def f_XtoY_competition(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, psi_1_2 = 0.0, psi_1_3 = 0.0, psi_2_3 = 0.0, F_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    log_sig1_k1 = np.log(1 + sig_1 / K_1)
    log_sig2_k2 = np.log(1 + sig_2 / K_2)
    log_sig3_k3 = np.log(1 + sig_3 / K_3)
    psi1_mult_log_sig1_k1 = psi_1 * log_sig1_k1
    psi2_mult_log_sig2_k2 = psi_2 * log_sig2_k2
    psi3_mult_log_sig3_k3 = psi_3 * log_sig3_k3
    F_1 = (
        psi1_mult_log_sig1_k1 * log_sig1_k1
        + psi2_mult_log_sig2_k2 * log_sig2_k2
        + psi3_mult_log_sig3_k3 * log_sig3_k3
    ) / (
        log_sig1_k1 + log_sig2_k2 + log_sig3_k3
    ) + (
        psi1_mult_log_sig1_k1 + psi2_mult_log_sig2_k2 + psi3_mult_log_sig3_k3
    )
    F_1 *= 0.5
    out = 1 / (1 + np.exp(-(F_1 + F_0)))
    return out

def f_XtoY_ret0(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, F_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions

    return 0.0

def memory_lifespan(T_eff, T_degrade = t_E_die, min_time = 0):

    out = np.maximum(t_long_M - (t_long_M - t_short_M)*T_eff/T_degrade, min_time)

    return out

def memory_protection(t, T_eff, T_degrade = t_E_die, target = N_0, min_time = 1):

    out = np.exp(- t/np.maximum(t_long_M - (t_long_M - t_short_M)*T_eff/T_degrade, min_time))

    return np.sum(out) - target

def antigenicity_over_harm(df):
    out = np.log(1 + (df['K_EH'] if 'K_EH' in df.columns.tolist() else K_EH)*(df['b_I']*(df['S_0'] if 'S_0' in df.columns.tolist() else S_0) - df['d_I'] + (df['d_H'] if 'd_H' in df.columns.tolist() else d_H) )/(df['d_I']*(df['I_0'] if 'I_0' in df.columns.tolist() else I_0)))/np.log(df['K_IE']/(df['I_0'] if 'I_0' in df.columns.tolist() else I_0))
    return out

def sigmoid(x, y_range, y_min,
            w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15,
            a0 = 0, a1 = 0, a2 = 0, a3 = 0):

    out = y_range \
    *1/(1 + np.exp( -((x[:,0])*w0 + (x[:,1])*w1 + (x[:,2])*w2 + (x[:,3])*w3 + a0))) \
    *1/(1 + np.exp( -((x[:,4])*w4 + (x[:,5])*w5 + (x[:,6])*w6 + (x[:,7])*w7 + a1))) \
    *1/(1 + np.exp( -((x[:,8])*w8 + (x[:,9])*w9 + (x[:,10])*w10 + (x[:,11])*w11 + a2))) \
    *1/(1 + np.exp( -((x[:,12])*w12 + (x[:,13])*w13 + (x[:,14])*w14 + (x[:,15])*w15 + a3))) \
    + y_min

    return out

def sigmoid_1d(x, y_range, y_min, b, w):

    out = y_range/(1 + np.exp(-w*x - b) ) + y_min
    return out

#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################

def lin_stoch_sim(S_0 = S_0, I_0 = I_0, b_I = b_I, d_S = d_S, d_I = d_I, d_IE = d_IE, K_IE = K_IE, K_SE = K_SE,
                  b_H = b_H, d_H = d_H, K_EH = K_EH,
                  N_0 = N_0, max_Na = max_Na, b_myc = b_myc, d_myc = d_myc, myc_thresh = myc_thresh, max_expand = max_expand,
                  char_times = [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_act],
                  NE_regulation = NE_psis,
                  EM_regulation = EM_psis,
                  activation_regulation = act_psis,
                  expansion_regulation = exp_psis,
                  infection_model = "acute",
                  duration = sim_duration,
                  steps = sim_steps,
                  reg_model = "mwc_like",
                  out_data = "small",
                  seed = None):
    rng = np.random.default_rng(seed)

    if reg_model == 'mwc_like':
        f_XtoY = f_XtoY_mwc_like
    elif reg_model == 'competition_model':
        f_XtoY = f_XtoY_competition
    else:
        f_XtoY = f_XtoY_ret0
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, K_IE := K_IE,
    # b_H := b_H, d_H := d_H, K_EH := K_EH,
    # N_0 := N_0, max_Na := max_Na, b_myc := b_myc, d_myc := d_myc, myc_thresh := myc_thresh,
    # char_times := [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die, t_Na],

    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_0*d_S

    # draw population of reponding cells for agent-based simulations
    psi_Na_I, psi_Na_H, psi_Na_P, F0_Na = activation_regulation[0], activation_regulation[1], activation_regulation[2], activation_regulation[3]
    psi_NE_I, psi_NE_H, psi_NE_P, F0_NE = NE_regulation[0], NE_regulation[1], NE_regulation[2], NE_regulation[3]
    psi_EM_I, psi_EM_H, psi_EM_P, F0_EM = EM_regulation[0], EM_regulation[1], EM_regulation[2], EM_regulation[3]
    psi_EE_I, psi_EE_H, psi_EE_P, F0_EE = expansion_regulation[0], expansion_regulation[1], expansion_regulation[2], expansion_regulation[3]

    p_tcr = 1.0 #np.ones(N_0_var)
    p_cyt = 1.0 #np.ones(N_0_var)

    int_steps = int(steps)

    # define variables for storage
    I = np.zeros(int_steps+1)
    S = np.zeros(int_steps+1)
    H = np.zeros(int_steps+1)
    P = np.zeros(int_steps+1)
    I_d_I = np.zeros(int_steps+1)
    I_d_IE = np.zeros(int_steps+1)
    S_d_SE = np.zeros(int_steps+1)

    N_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    N_m[0,:] +=1
    T_Ecyt = [[] for i in np.arange(0, N_0_var)]
    # T_Ecyt = np.zeros(int_steps+1, dtype = np.int32)

    div_count = np.ones(N_0_var)
    div_E_count = np.ones(N_0_var)
    div_M_count = np.ones(N_0_var)
    diff_EpM_count = np.zeros(N_0_var, dtype = np.int32)

    Na_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    Ma_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    E_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32) # effector in periphary
    T_E = np.zeros(N_0_var)

    if out_data == "full":
        mycN_m = np.zeros((int_steps+1, N_0_var))
        mycMa_m = np.zeros((int_steps+1, N_0_var))
        mycE_m = np.zeros((int_steps+1, N_0_var))

    bias_t = np.zeros((int_steps+1, 4))

    # Define event timer variables
    bound_N = np.zeros(N_0_var, dtype =np.int32)
    bound_N_time = np.zeros(N_0_var)
    Na_div_flag = np.ones(N_0_var, dtype =np.int32)
    Na_flag = np.ones(N_0_var, dtype =np.int32)

    mycN = np.zeros(N_0_var)
    mycE = np.zeros(N_0_var)
    mycMa = np.zeros(N_0_var)

    r_NaE = np.zeros((int_steps+1, N_0_var))
    r_EM = np.zeros((int_steps+1, N_0_var))
    r_Na = np.zeros((int_steps+1, N_0_var))
    r_EE = np.zeros((int_steps+1, N_0_var))
    cum_r_NaE = np.zeros(N_0_var)

    b_unbind_t = np.zeros(N_0_var)
    b_N_bind = np.zeros(N_0_var)
    b_Na_div = np.ones(N_0_var)/char_times[2]
    b_myc_t = np.zeros(N_0_var)
    b_E_div = np.ones(N_0_var)/char_times[3]
    b_Ma_div = np.ones(N_0_var)/char_times[4]
    d_E_die = np.ones(N_0_var)/char_times[5]

    #################################
    ### RUN POPULATION SIMULATION ###
    #################################
    t = 0.0
    S[0] = S_0
    I[0] = I_0
    H[0] = 0.0
    P[0] = 0.0

    sum_e_m = 0

    for i in np.arange(1, int_steps + 1):
        # select virulence model:
        if (b_I > 0 and I_0 <= S_0/1000) or infection_model == 'acute': # "acute"
            b_I_t = b_I
            d_Sauto = 0.0
            infection_model = 'acute'
        elif (b_I > 0 and I_0 > S_0/1000) or infection_model == 'cancer': # "cancer"
            b_I_t = b_I/S_0
            d_Sauto = 0.0
            infection_model = 'cancer'
        elif b_I == 0 or infection_model == 'autoimmune': # "autoimmune"
            b_I_t = 0.0
            d_Sauto = d_I
            K_SE = K_IE
            I_0 = 0.0
            infection_model = 'autoimmune'
        else:
            b_I_t = b_I
            d_Sauto = 0

        # Compute total population of cell types
        # Use sum computed in previous iteration.
        E_pop = sum_e_m

        #### Run infection dynamics: replication and effector clearance ####

        # Update state of susceptible, infected and inflammation
        # (a) event variables
        S_to_I = dt * S[i - 1] * (I[i - 1] >= I_0) * b_I_t * I[i - 1] * (S[i - 1] >= 1.0)
        I_d_IE[i] += I_d_IE[i - 1] + dt * I[i - 1] * d_IE * E_pop / (K_IE + I[i - 1] + E_pop) # infected/cancer cells killed by immune response
        I_d_I[i] += I_d_I[i - 1] + dt * I[i - 1]*d_I*(infection_model == "acute") # cells killed by infection
        S_d_SE[i] += S_d_SE[i - 1] + dt * S[i - 1] * (d_IE * E_pop / (K_SE + S[i - 1] + E_pop)) # susceptible cells killed by immune response

        # (b) state variables
        S[i] += S[i - 1] - S_to_I + dt * (
            b_S - (d_S + (d_Sauto / t_Hauto) * np.exp(-(t / t_Hauto)**2 / 2)) * S[i - 1]
            - (S_d_SE[i] - S_d_SE[i - 1])
        ) * (S[i - 1] >= 1.0)

        I[i] += I[i - 1] + S_to_I - (I_d_IE[i] + I_d_I[i] - I_d_IE[i - 1] - I_d_I[i - 1])

        harm_detected = (I_d_I[i] - I_d_I[i - 1]) + dt * S[i - 1] * (d_Sauto / t_Hauto) * np.exp(-(t / t_Hauto)**2 / 2) + S_to_I * (infection_model == "cancer")
        immunopathology = (S_d_SE[i] - S_d_SE[i - 1]) + (I_d_IE[i] - I_d_IE[i - 1])
        cells_sensed = I[i - 1] + (S[i - 1] * K_IE / K_SE)

        H[i] += H[i - 1] + b_H * harm_detected - dt * d_H * H[i - 1] * (H[i - 1] >= 0.0)
        
        P[i] += P[i - 1] + b_H * immunopathology - dt * d_H * P[i - 1] * (P[i - 1] >= 0.0)

        ## I. Recruitment/Priming

        # (a) Phase 1: Naive cells encounter and bind APCs
        if cells_sensed >= 1:
            bound_N += rng.binomial(N_m[i - 1], b_N_bind * dt) - rng.binomial(N_m[i - 1], b_unbind_t * dt)
        else:
            bound_N = 0

        bound_N_time += dt * (bound_N - (1 - bound_N) * bound_N_time * N_m[i - 1]) # only reset to zero if the cell is not activated

        # (b) Phase 2: Activated naive cells are bound to APCs and receive stimulation.
        act_N = (1 - bound_N) * (mycN >= myc_thresh) * (N_m[i - 1] > 0)

        # (c) Phase 3: Unbound activated naive cells divide and then differentiate
        div_Na = rng.binomial(Na_m[i - 1] * Na_div_flag, dt * b_Na_div)

        diff_NaE = rng.binomial(
            Na_m[i - 1] * (1 - Na_div_flag),
            1 - np.exp(-cum_r_NaE*dt) if np.sum(Na_m[i - 1]) > 0 else 0
        )

        ## II. Expansion

        # (a) New central memory cells divide
        div_Ma = rng.binomial(Ma_m[i - 1] - diff_EpM_count, dt * b_Ma_div)

        # (b) Effector cells divide, differentiate, die
        die_E = rng.binomial(E_m[i - 1], d_E_die * dt)
        div_E = rng.binomial(E_m[i - 1] - die_E, dt * b_E_div)
        diff_EM = rng.binomial(E_m[i - 1] - die_E, dt * r_EM[i - 1])

        #### Update population dynamics: ####
        N_m[i] += N_m[i - 1] - act_N
        Na_m[i] += Na_m[i - 1] + div_Na + act_N - (1 - Na_div_flag) * Na_m[i - 1]

        Ma_m[i] += Ma_m[i - 1] + div_Ma + (1 - Na_div_flag)*(Na_m[i - 1] - diff_NaE) + diff_EM # should I program memory death
        E_m[i] += E_m[i - 1] + div_E + (1 - Na_div_flag) * (diff_NaE) - (die_E + diff_EM)

        # Evaluate sums and create masks
        sum_n_m = np.sum(N_m[i])
        sum_e_m = np.sum(E_m[i])
        e_m_nonzero_mask = E_m[i] > 0
        n_m_nonzero_mask = N_m[i] > 0
        na_m_nonzero_mask = Na_m[i] > 0
        n_na_sum_nonzero_mask = n_m_nonzero_mask | na_m_nonzero_mask

        # Update division flag to allow division to proceed
        Na_div_flag = (Na_m[i] < max_Na)*(na_m_nonzero_mask)
        div_count += div_Na + div_E + div_Ma
        div_E_count += div_E
        div_M_count += div_Ma
        diff_EpM_count += diff_EM

        #### New binding events ####
        b_N_bind = (n_m_nonzero_mask) * (1 - bound_N) * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = 0.0, psi_2 = psi_max, psi_3 = 0.0, F_0 = -psi_max*np.log(2), K_1 = S_0, K_2 = K_EH, K_3 = K_EH
        ) / (char_times[0] if infection_model != "autoimmune" else t_bind_eM) if cells_sensed >= 1 else 0.0

        b_unbind_t = bound_N * np.fmin(
            2 * bound_N_time / (
                f_XtoY(
                    sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
                    psi_1 = psi_max, psi_2 = 0.0,psi_3 = 0.0, F_0 = -psi_max*np.log(2),
                    K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
                    K_2 = K_EH, K_3 = K_EH
                ) * char_times[1] * 2 / np.sqrt(np.pi)
            )**2, 1 / dt
        ) if np.sum(bound_N) >= 1 else 0.0

        b_myc_t = b_myc * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = psi_Na_I, psi_2 = psi_Na_H, psi_3 = psi_Na_P, F_0 = F0_Na,
            K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
            K_2 = K_EH, K_3 = K_EH
        ) * bound_N

        #### MYC Dynamics ####
        mycN = (mycN + dt * (b_myc_t - (1 - bound_N) * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = -psi_EE_I, psi_2 = -psi_EE_H, psi_3 = -psi_EE_P, F_0 = -F0_EE,
            K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
            K_2 = K_EH, K_3 = K_EH
        ) * mycN * d_myc * (mycN > 0))) * n_na_sum_nonzero_mask # (1 - bound_N)*

        mycE = (mycE - dt * (f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = -psi_EE_I, psi_2 = -psi_EE_H, psi_3 = -psi_EE_P, F_0 = -F0_EE,
            K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
            K_2 = K_EH, K_3 = K_EH)*mycE*d_myc)
        ) * (E_m[i] >= 1) + mycN * (na_m_nonzero_mask)

        mycMa = (mycMa - dt * mycMa * d_myc) * (Ma_m[i] > 0) + mycN * na_m_nonzero_mask # higher decay rate of myc

        #### Time-dependent rates modulated by antigen and cytokine signals ####
        b_Na_div = 1 / char_times[2]
        b_E_div = (mycE >= myc_thresh) * (div_count < max_expand) / char_times[3]
        b_Ma_div = (mycMa >= myc_thresh) * (div_count < max_expand) / char_times[4]

        #### Transition probabilities modulated by antigen and cytokine signals ####
        r_Na[i] += b_myc_t / myc_thresh

        r_NaE[i] += (r_NaE[i - 1] + 2 * (mycN >= myc_thresh) * dt * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = psi_NE_I, psi_2 = psi_NE_H, psi_3 = psi_NE_P,F_0 = F0_NE,
            K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
            K_2 = K_EH, K_3 = K_EH
        ) / (2 / np.sqrt(np.pi) * (char_times[1] - char_times[6]))**2) * n_na_sum_nonzero_mask

        r_EM[i] += (r_EM[i - 1] + 2 * dt * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*H[i], sig_3 = p_cyt*P[i],
            psi_1 = psi_EM_I, psi_2 = psi_EM_H, psi_3 = psi_EM_P, F_0 = F0_EM,
            K_1 = (K_IE*(infection_model != "autoimmune") + K_SE*(infection_model == "autoimmune")),
            K_2 = K_EH, K_3 = K_EH
        ) / (2 / np.sqrt(np.pi) * (char_times[1] + char_times[6]))**2) * e_m_nonzero_mask # char_times[6] + char_times[6] to reverse priming and differentiation

        r_EE[i] += b_E_div

        cum_r_NaE += r_NaE[i]

        # store time cells become effector
        T_E += dt * e_m_nonzero_mask if sum_e_m > 0 else 0.0
        d_E_die = 0.5 * np.sqrt(np.pi) * e_m_nonzero_mask * T_E / char_times[5]**2

        # store time an effector spends in cytotoxic state
        for_where = (diff_NaE + div_Ma + diff_EM > 0)
        where_output = np.where(for_where)[0]
        for l in where_output:
            T_Ecyt[l] += (diff_NaE[l] + div_Ma[l]) * [0.0] + diff_EM[l] * [T_E[l].item()]

        #### Store myc levels ####
        if out_data == "full":
            mycN_m[i] += mycN
            mycMa_m[i] += mycMa
            mycE_m[i] += mycE

        #### Store differentiation biases
        to_add_to_bias = np.zeros(4)
        if sum_n_m > 0.:
            to_add_to_bias[0] = np.mean(r_Na[i][n_m_nonzero_mask])
            to_add_to_bias[1] = np.mean(r_NaE[i][n_m_nonzero_mask])
        if sum_e_m > 0.:
            to_add_to_bias[2] = np.mean(r_EM[i][e_m_nonzero_mask])
            to_add_to_bias[3] = np.mean(r_EE[i][e_m_nonzero_mask])
        bias_t[i] += to_add_to_bias

        # Increment time
        t += dt

    # Collect population dynamics
    N, Na, Ma, E = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(Ma_m, axis = 1), np.sum(E_m, axis = 1)

    T_pEcytM = np.concatenate(T_Ecyt) if N_0 > 0 else np.array([0.0]) # time that memory spends in effector
    count_cM = np.sum(T_pEcytM == 0
                     ) if len(T_pEcytM[T_pEcytM == 0]) > 0 and N_0 > 0.0 else 0.0
    count_eM = np.sum(memory_lifespan(T_pEcytM) > char_times[5]
                     ) if len(T_pEcytM[memory_lifespan(T_pEcytM) > char_times[5]]) > 0 and N_0 > 0.0 else 0.0

    # Project out surviving primary memory
    cM_duration = np.mean(memory_lifespan(T_pEcytM[T_pEcytM == 0])
                         )/365.25 if len(T_pEcytM[T_pEcytM == 0]) > 0 and N_0 > 0.0 else 0.0
    eM_duration = np.mean(memory_lifespan(T_pEcytM[memory_lifespan(T_pEcytM) > char_times[5]])
                         )/365.25 if len(T_pEcytM[memory_lifespan(T_pEcytM) > char_times[5]]) > 0 and N_0 > 0.0 else 0.0

    zeros_n_0_var = np.zeros(N_0_var)

    lineage_comp = np.vstack([
        np.amax(Na_m, axis = 0), div_M_count, div_E_count,
        p_tcr + zeros_n_0_var, p_cyt + zeros_n_0_var
    ])

    dyn_data = np.array([S, I, N, Na, E, Ma, H, I_d_I + I_d_IE, S_d_SE, P]).T # add I_d_IE as a separate variable
    prim_bias = bias_t

    ts = np.linspace(0, duration, int_steps + 1)

    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, pI, N, Na, pE, pM, pH, pI_d_I, pI_d_SE, pP = dyn_data[:,0], dyn_data[:,1], dyn_data[:,2], dyn_data[:, 3], dyn_data[:,4], dyn_data[:,5], dyn_data[:,6], dyn_data[:,7], dyn_data[:,8], dyn_data[:,9]

    dt = ts[1]-ts[0]

    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, K_IE,
              b_H, d_H, K_EH,
              N_0, max_Na, b_myc, d_myc, myc_thresh]),
              char_times,
              activation_regulation, NE_regulation, EM_regulation, expansion_regulation))

    sim_summary = np.array([np.sum(pI*dt) + np.sum(pS*dt)*(infection_model == "autoimmune"),
                       np.argmax(pI)*dt,
                       np.argmax(pI < I_0)*dt,
                       np.amax(pI_d_I) + I[-1],
                       np.amax(pI_d_SE),
                       np.sum(div_E_count),
                       np.argmax(pE)*dt + sim_duration*(np.max(pE) < 1.0),
                       (np.argmax(pE >= N_0 , axis = 0)*dt + sim_duration*(np.max(pE) < N_0)) if N_0 > 0 else sim_duration,
                       count_eM,
                       eM_duration,
                       count_cM,
                       cM_duration,
                       np.sum(pP*dt),
                       np.sum(pH*dt),
                       np.amin(pS)])

    # down-size timeseries
    pnts = int(0.1*steps)
    keep = [i*10 for i in np.arange(0,pnts)]

    if out_data == "full":
        out_dict = {"reg_coeffs": np.concatenate((activation_regulation, NE_regulation, EM_regulation, expansion_regulation)), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "eff_by_lin": (E_m), "N_myc_by_lin": mycN_m, "Ma_myc_by_lin": mycMa_m, "E_myc_by_lin": mycE_m, "parameters": parameters, "summary_stats": sim_summary, "pmemory_formed": T_pEcytM if N_0 > 0 else []}
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
              r"$(eM)^{max}$",
              r"$T_{eM < N}$",
              r"$(cM)^{max}$",
              r"$T_{cM < N}$",
              r"$\int_0^{T_{sim}} P_p dt$",
              r"$\int_0^{T_{sim}} H_p dt$",
              r"$S_p^{min}$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$b_H$", r"$d_H$", r"$K_{E,I}$", r"$K_{E,H}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N,A_{in}}$", r"$\tau_{N \cdot A_{in}}$", r"$\tau_{N^*,N^*}$", r"$\tau_{E_,E}$", r"$\tau_{M,M}$", r"$\tau_{E_{die}}$", r"$\tau_{N^*}$",
               r"$\psi_{N^*}^{(I)}$", r"$\psi_{N^*}^{(H)}$", r"$\psi_{N^*}^{(P)}$", r"$F_{N^*}$",
               r"$\psi_{N,E}^{(I)}$", r"$\psi_{N,E}^{(H)}$", r"$\psi_{N,E}^{(P)}$", r"$F_{N^{*},E}$",
               r"$\psi_{E,M}^{(I)}$", r"$\psi_{E,M}^{(H)}$", r"$\psi_{E,M}^{(P)}$", r"$F_{E,M}$",
               r"$\psi_{E^*}^{(I)}$", r"$\psi_{E^*}^{(H)}$", r"$\psi_{E^*}^{(P)}$", r"$F_{E^*}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'K_IE',
                      'b_H', 'd_H', 'K_EH',
                      'N_0', 'max_Na', 'b_myc', 'd_myc', 'myc_thresh',
                      't_bind', 't_unbind', 't_Na_div', 't_E_div', 't_M_div', 't_E_die', 't_act',
                      'psi_Na_I', 'psi_Na_H', 'psi_Na_P', 'F0_Na',
                      'psi_NE_I', 'psi_NE_H', 'psi_NE_P', 'F0_NE',
                      'psi_EM_I', 'psi_EM_H', 'psi_EM_P', 'F0_EM',
                      'psi_EE_I', 'psi_EE_H', 'psi_EE_P', 'F0_EE']

stat_names_for_df = ['p_load', 'T_max_pI', 'T_min_pI', 'harm_pI',
                     'harm_pS', 'max_pE', 'T_pE_max', 'T_pE_start',
                     'max_eM', 'T_eM_min', 'max_cM', 'T_cM_min',
                     'int_pP', 'int_pH', 'min_pS']

Na_reg = ['psi_Na_I', 'psi_Na_H', 'psi_Na_P', 'F0_Na']
NE_reg = ['psi_NE_I', 'psi_NE_H', 'psi_NE_P', 'F0_NE']
EM_reg = ['psi_EM_I', 'psi_EM_H', 'psi_EM_P', 'F0_EM']
EE_reg = ['psi_EE_I', 'psi_EE_H', 'psi_EE_P', 'F0_EE']

module_labels = ['$N \longrightarrow N^*$', '$N^* \longrightarrow E$', '$E \longrightarrow M$', '$E \longrightarrow E + E$']
modules = ['Na', 'NE', 'EM', 'EE']
reg_stim = Na_reg[0:3] + NE_reg[0:3] + EM_reg[0:3] + EE_reg[0:3]
reg_bl = [Na_reg[3]] + [NE_reg[3]] + [EM_reg[3]] + [EE_reg[3]]
reg_bl_label = [param_names[-13]] + [param_names[-9]] + [param_names[-5]] + [param_names[-1]]

perf_vars = ["peff_protection", "peff_toxicity", "max_cM_fold", "max_eM_fold", "T_eM_min"] # , "max_pE_fold", "T_max_pI", "T_pE_start", 'T_pE_clear']
perf_labels = ["Protection", "Toxicity", "C. memory exp.", "E. memory exp.", "E. memory dur."] # "Response expansion", "Clear. timing \n (days)", "Resp. timing \n (days)", "Resp. clear \n (days)"]

key_var = 'antigenicity_over_harm'
key_var_label = "Ag.-inflam. salience\n"+r"$\frac{\tau_I^{-1}}{\tau_H^{-1}}$"

vir_vars = ['d_I', 'K_IE', 'b_I', 'K_EH', 'N_0', 'S_0', 'I_0']