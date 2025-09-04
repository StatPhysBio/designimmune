### Stochastic model dynamics ###
import numpy as np
import itertools
from scipy.optimize import fsolve, minimize

### (1) Define simulation parameters
# Define simulation hyperparameters
sim_duration = 30
sim_steps = int(0.3*(10**4))

# infection dynamics
S_max = 10**7 # max susceptible cells
d_S = 0.01 # susceptible cell death rate
b_I = 1/S_max # fecudity of pathogen (Chao et al. 2004, Iwami et al. 2015)
I_min = 10**(-4)*S_max # initial detectable level of infected cells
d_IE = 12 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_I = 10**(-3)*S_max # max effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
d_I = 0.5 # successful virus cannot kill cells faster than it infects new ones
K_S = S_max # effector avidity (half-max) for susceptible cells

# Inflammatory response
d_H = 1.0 # decay of cytokine response
K_H = 10**(-3)*S_max # half-max level of innate/inflammatory response required to trigger lymphocyte response

# Immune cells
N_0 = 100 # initial number of naive cells
max_Na = 2**2 # number of signal-independent divisions after activation
max_expand = 2**15 # maximum clone size (Marchingo et al.)
t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_cycle = 1.0, 1.0, 8.8/24, 8/24, 12/24, 10.0, 1/4
t_long_M = 10*365.25 # lifespan of central memory cells
t_short_M = 365.25/12 # lifespan of effector memory cells

# Auto-immunity and cancer
t_Hauto = 1.0 # duration of autoimmune inflammation
b_C = 1/10 # growth rate of cancer

# Pathogen space parameters
num_pnts = 9
infection_sample = np.array(np.meshgrid(np.array([d_I]), # vary d_I
                                   S_max*np.logspace(-3.0, 0, num_pnts), # vary K_I
                                   b_I*np.linspace(0.5, 2.5, num_pnts), # vary b_I
                                   K_H*np.array([1.0]), # vary K_H
                                   N_0*np.array([1.0]), # vary N_0
                                   I_min*np.logspace(0.0, 0.0, 1), # vary I_0
                                   np.array([S_max]) # vary K_S
                                           )).T.reshape(-1,7)

auto_sample = np.array(np.meshgrid(d_S*np.array([1.0]), # vary d_I
                                   K_S*np.logspace(-1.0, 0, 3), # vary K_I
                                   b_I*np.array([0.0]), # vary b_I
                                   K_H*np.array([1.0]), # vary K_H
                                   N_0*np.logspace(-1.0, 1, 5), # vary N_0
                                   I_min*np.array([0.0]) # vary I_0
                                           )).T.reshape(-1,6)

cancer_sample = np.array(np.meshgrid(d_S*np.array([1.0]), # vary d_I
                                   S_max*np.logspace(-1.0, 1.0, 3), # vary K_I
                                   b_C*np.array([1.0]), # vary b_I
                                   K_H*np.array([1.0]), # vary K_H
                                   N_0*np.logspace(-1.0, 1, 3), # vary N_0
                                   S_max*np.array([0.05]), # vary I_0
                                   np.array([S_max]) # vary K_S
                                           )).T.reshape(-1,7)

infection_sample_select = np.vstack([infection_sample,
                                     #auto_sample,
                                     #cancer_sample
                                    ])

# Design space hyper parameters
pnts = 13
psi_max = 3.0
L0_max = 3.0
psi_4d = np.array(list(itertools.product(np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                 np.linspace(-L0_max, L0_max, int(pnts)).tolist())))

psi_6d = np.array(list(itertools.product(np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                         np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                         np.linspace(-psi_max, psi_max, int(pnts)).tolist(),
                                         np.linspace(-L0_max, L0_max, int(pnts)).tolist(),
                                         np.linspace(-L0_max, L0_max, int(pnts)).tolist(),
                                         np.linspace(-L0_max, L0_max, int(pnts)).tolist()
                                        )))

psi_sparse = np.array([psi_6d[:,0], psi_6d[:,1], psi_6d[:,2], psi_6d[:,3], # Na
          psi_6d[:,0], psi_6d[:,1], psi_6d[:,2], psi_6d[:,4], # NE
          -0*psi_6d[:,0], -0*psi_6d[:,1], -0*psi_6d[:,2],-L0_max**4 -0*psi_6d[:,4], # EM
          -psi_6d[:,0], -psi_6d[:,1], -psi_6d[:,2], psi_6d[:,5]]).T  # Edie

# default regulatory weights:
act_psis = [psi_max, psi_max, -psi_max, L0_max]
NE_psis = [psi_max, psi_max, -psi_max, L0_max]
EM_psis = [-0*psi_max, -0*psi_max, 0*psi_max, -L0_max**4]
contract_psis = [psi_max, psi_max, -psi_max, 0.0]

### (2) Define functions for simulations
# Define functions for simulations

def f_XtoY_mwc_like(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, L_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # L_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    L_1 = (
        psi_1 * np.log(1 + sig_1 / K_1)
        + psi_2 * np.log(1 + sig_2 / K_2)
        + psi_3 * np.log(1 + sig_3 / K_3)
    )
    out = 1 / (1 + np.exp(-(L_1 + L_0)))
    return out

def f_XtoY_ret0(sig_1 = 0.0, sig_2 = 0.0, sig_3 = 0.0, psi_1 = 0.0, psi_2 = 0.0, psi_3 = 0.0, L_0 = 0.0, K_1 = 1.0, K_2 = 1.0, K_3 = 1.0):
    ### variable
    # sig_. := signal .
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # L_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions

    return 0.0

def memory_lifespan(T_eff, T_degrade = t_E_die, min_time = 0):
    # out =  np.maximum(t_long_M - (t_long_M - t_short_M)*T_eff/T_degrade, min_time)
    out = t_long_M*np.exp(-T_eff/T_degrade*np.log(t_long_M/t_short_M))
    return out

def antigenicity_over_harm(df):
    out = np.log(1 + (df['K_H'] if 'K_H' in df.columns.tolist() else K_H)*(df['b_I']*(df['S_max'] if 'S_max' in df.columns.tolist() else S_max) - df['d_I'] + (df['d_H'] if 'd_H' in df.columns.tolist() else d_H) )/(df['d_I']*(df['I_0'] if 'I_0' in df.columns.tolist() else I_0)))/np.log(df['K_I']/(df['I_0'] if 'I_0' in df.columns.tolist() else I_min))
    return out

def sigmoid(x, y_max, y_min,
            w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
            a0 = 0, a1 = 0, a2 = 0, a3 = 0):

    out = (y_max - y_min) \
    *1/(1 + np.exp( -((x[:,0])*w0 + (x[:,1])*w1 + (x[:,2])*w2 + (x[:,3])*w3 + a0))) \
    *1/(1 + np.exp( -((x[:,4])*w4 + (x[:,5])*w5 + (x[:,6])*w6 + (x[:,7])*w7 + a1))) \
    *1/(1 + np.exp( -((x[:,8])*w8 + (x[:,9])*w9 + (x[:,10])*w10 + (x[:,11])*w11 + a2))) \
    + y_min

    return out

def modified_sigmoid(x, y_max, y_min,
                     w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                     a0 = 0, a1 = 0, a2 =0, a3 =0,
                     b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0, b8 = 0):

    out = (y_max - y_min) \
    *1/(1 + np.exp( -((x[:,0])*w0 + (x[:,1])*w1 + (x[:,2])*w2 + (x[:,3])*w3 + a0 + b0*(x[:,0]*x[:,1]) + b1*(x[:,0]*x[:,2]) + b2*(x[:,1]*x[:,2]) ))) \
    *1/(1 + np.exp( -((x[:,4])*w4 + (x[:,5])*w5 + (x[:,6])*w6 + (x[:,7])*w7 + a1 + b3*(x[:,4]*x[:,5]) + b4*(x[:,4]*x[:,6]) + b5*(x[:,5]*x[:,6]) ))) \
    *1/(1 + np.exp( -((x[:,8])*w8 + (x[:,9])*w9 + (x[:,10])*w10 + (x[:,11])*w11 + a2 + b6*(x[:,8]*x[:,9]) + b7*(x[:,8]*x[:,10]) + b8*(x[:,9]*x[:,10]) ))) \
    + y_min

    return out

def sigmoid_1d(x, y_max, y_min, b, w):

    out = (y_max - y_min)/(1 + np.exp(-w*x - b) ) + y_min
    return out

#######################
## AGENT-BASED STOCHASTIC SIMULATION WITH TAU-LEAPING
#######################

def lin_stoch_sim(S_0 = S_max - I_min, I_0 = I_min, b_I = b_I, d_S = d_S, d_I = d_I, d_IE = d_IE, K_I = K_I, K_S = K_S,
                  d_H = d_H, K_H = K_H,
                  N_0 = N_0, max_Na = max_Na, max_expand = max_expand,
                  char_times = [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_E_die, t_cycle],
                  NE_regulation = NE_psis,
                  EM_regulation = EM_psis,
                  activation_regulation = act_psis,
                  contraction_regulation = contract_psis,
                  infection_model = "acute",
                  duration = sim_duration,
                  steps = sim_steps,
                  reg_model = "mwc_like",
                  out_data = "small",
                  memory_death = False,
                  seed = None):
    rng = np.random.default_rng(seed)

    if reg_model == 'mwc_like':
        f_XtoY = f_XtoY_mwc_like
    else:
        f_XtoY = f_XtoY_ret0
        
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, K_I := K_I,
    # d_H := d_H, K_H := K_H,
    # N_0 := N_0, max_Na := max_Na,
    # char_times := [t_bind, t_unbind, t_Na_div, t_E_div, t_M_div, t_EM_diff, t_E_die, t_Na],

    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_max*d_S

    # draw population of reponding cells for agent-based simulations
    psi_myc_I, psi_myc_HI, psi_myc_HE, L0_Na = activation_regulation[0], activation_regulation[1], activation_regulation[2], activation_regulation[3]
    psi_NE_I, psi_NE_HI, psi_NE_HE, L0_NE = NE_regulation[0], NE_regulation[1], NE_regulation[2], NE_regulation[3]
    psi_EM_I, psi_EM_HI, psi_EM_HE, L0_EM = EM_regulation[0], EM_regulation[1], EM_regulation[2], EM_regulation[3]
    psi_Edie_I, psi_Edie_HI, psi_Edie_HE, L0_Edie = contraction_regulation[0], contraction_regulation[1], contraction_regulation[2], contraction_regulation[3]

    p_tcr = 1.0 #np.linspace(1/10, 1, N_0_var)
    p_cyt = 1.0 #np.ones(N_0_var)

    int_steps = int(steps)

    # define variables for storage
    I = np.zeros(int_steps+1)
    S = np.zeros(int_steps+1)
    HI = np.zeros(int_steps+1)
    HE = np.zeros(int_steps+1)
    I_d_I = np.zeros(int_steps+1)
    I_d_E = np.zeros(int_steps+1)
    S_d_E = np.zeros(int_steps+1)

    N_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    N_m[0,:] += 1
    T_Ecyt = [[] for i in np.arange(0, N_0_var)]
    survive_M = [[] for i in np.arange(0, N_0_var)]

    div_count = np.zeros(N_0_var)
    div_E_count = np.log2(max_Na)*np.ones(N_0_var)
    div_M_count = np.log2(max_Na)*np.ones(N_0_var)
    diff_EpM_count = np.zeros(N_0_var, dtype = np.int32)
    die_M = np.zeros(N_0_var, dtype = np.int32)

    Na_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    Ma_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32)
    E_m = np.zeros((int_steps+1, N_0_var), dtype = np.int32) # effector in periphary
    T_E = np.zeros(N_0_var)

    bias_t = np.zeros((int_steps+1, 4))

    # Define event timer variables
    bound_N = np.zeros(N_0_var, dtype =np.int32)
    bound_N_time = np.zeros(N_0_var)
    Na_div_flag = np.ones(N_0_var, dtype =np.int32)
    Na_flag = np.ones(N_0_var, dtype =np.int32)
    act_N = np.zeros(N_0_var, dtype =np.int32)

    r_NaE = np.zeros((int_steps+1, N_0_var))
    r_EM = np.zeros((int_steps+1, N_0_var))
    r_Na = np.zeros((int_steps+1, N_0_var))
    r_E_div = np.zeros((int_steps+1, N_0_var))
    r_Edie = np.zeros((int_steps+1, N_0_var))
    cum_r_NaE = np.zeros(N_0_var)

    r_unbind_t = np.zeros(N_0_var)
    r_N_bind = np.zeros(N_0_var)
    r_Na_div = np.ones(N_0_var)/char_times[2]
    r_Ma_div = np.ones(N_0_var)/char_times[4]

    #################################
    ### RUN POPULATION SIMULATION ###
    #################################
    t = 0.0
    S[0] = S_0
    I[0] = I_0
    HI[0] = 0.0
    HE[0] = 0.0

    sum_e_m = 0

    for i in np.arange(1, int_steps + 1):
        # select virulence model:
        if 'acute' in infection_model : # "acute"
            b_I_t = b_I
            d_Sauto = 0.0
            infection_model = 'acute'
            
        elif 'cancer' in infection_model: # "cancer"
            b_I_t = b_I*(1 - I[i - 1]/S_max)/S[i - 1]
            d_Sauto = 0.0
            infection_model = 'cancer'
            
        elif b_I == 0 or 'autoimmune' in infection_model: # "autoimmune"
            b_I_t = 0.0
            d_Sauto = d_I
            K_S = K_I
            I_0 = 0.0
            infection_model = 'autoimmune'
            
        else:
            b_I_t = b_I
            d_Sauto = 0
            infection_model = 'acute'

        # Compute total population of cell types
        # Use sum computed in previous iteration.
        E_pop = sum_e_m

        #### Run infection dynamics: replication and effector clearance ####

        # Update state of susceptible, infected and inflammation
        # (a) Record new cell infections and cell killing events
        S_to_I = dt * S[i - 1] * b_I_t * I[i - 1] * (S[i - 1] >= 1.0) * (I[i - 1] >= I_min) * (I[i - 1] <= S_max)
        I_d_E[i] += I_d_E[i - 1] + dt * I[i - 1] * d_IE * E_pop / (K_I + S[i - 1]*(K_I + E_pop)/(K_S + E_pop) + I[i - 1] + E_pop) # infected/cancer cells killed by immune response
        I_d_I[i] += I_d_I[i - 1] + dt * I[i - 1]*d_I*(infection_model == "acute") + S_to_I * (infection_model == "cancer") # cells killed by infection or cancer
        S_d_E[i] += S_d_E[i - 1] + dt * S[i - 1] * (d_IE * E_pop / (K_S + I[i - 1]*(K_S + E_pop)/(K_I + E_pop) + S[i - 1] + E_pop)) # susceptible cells killed by immune response

        # (b) Evolve susceptible and infected populations
        S[i] += S[i - 1] - S_to_I + dt * (
            0*b_S - (0*d_S + (d_Sauto / t_Hauto) * np.exp(-(t / t_Hauto)**2 / 2)) * S[i - 1]
            - (S_d_E[i] - S_d_E[i - 1])/dt
        ) * (S[i - 1] >= 1.0)

        I[i] += I[i - 1] + S_to_I - (I_d_E[i] + I_d_I[i] - I_d_E[i - 1] - I_d_I[i - 1]) + S_to_I * (infection_model == "cancer")

        # (C) Evolve inflammatory responses
        harm_detected = (I_d_I[i] - I_d_I[i - 1]) + dt * S[i - 1] * (d_Sauto / t_Hauto) * np.exp(-(t / t_Hauto)**2 / 2)
        immunopathology = (S_d_E[i] - S_d_E[i - 1]) + (I_d_E[i] - I_d_E[i - 1])
        cells_sensed = I[i - 1] # + (S[i - 1] * K_I / K_S)

        HI[i] += HI[i - 1] + harm_detected - dt * d_H * HI[i - 1] * (HI[i - 1] >= 0.0)
        
        HE[i] += HE[i - 1] + immunopathology - dt * d_H * HE[i - 1] * (HE[i - 1] >= 0.0)

        ## I. Recruitment/Priming

        # (a) Phase 1: Naive cells encounter and bind APCs
        if cells_sensed >= 1:
            bound_N += rng.binomial(N_m[i - 1], r_N_bind * dt) - act_N #- rng.binomial(N_m[i - 1], r_unbind_t * dt)
        else:
            bound_N = 0

        bound_N_time += dt * bound_N - (1 - bound_N) * bound_N_time * N_m[i - 1] # only reset to zero if the cell is not activated

        # (b) Phase 2: Activated naive cells are bound to APCs and receive stimulation.
        act_N = rng.binomial(N_m[i - 1], dt * r_Na[i - 1])

        ## II. Early differentiation

        # (a) Unbound activated naive cells divide and then differentiate
        div_Na = rng.binomial(Na_m[i - 1] * Na_div_flag, dt * r_Na_div)

        diff_NaE = rng.binomial(
            Na_m[i - 1] * (1 - Na_div_flag),
            1 - np.exp(-cum_r_NaE * dt) if np.sum(Na_m[i - 1]) > 0 else 0
        )

        ## III. Expansion and contraction

        # (a) New central memory cells divide
        div_Ma = rng.binomial(Ma_m[i - 1] - diff_EpM_count, dt * r_Ma_div)
        
        # (b) Effector cells divide, differentiate, die
        die_E = rng.binomial(E_m[i - 1], dt * r_Edie[i - 1])
        diff_EM = rng.binomial(E_m[i - 1] - die_E, dt * r_EM[i - 1])
        div_E = rng.binomial(E_m[i - 1] - die_E - diff_EM, dt * r_E_div[i - 1])
        diff_EpM_count += diff_EM - die_M

        #### Update population dynamics: ####
        N_m[i] += N_m[i - 1] - act_N
        Na_m[i] += Na_m[i - 1] + div_Na + act_N - (1 - Na_div_flag) * Na_m[i - 1]

        Ma_m[i] += Ma_m[i - 1] + div_Ma + (1 - Na_div_flag)*(Na_m[i - 1] - diff_NaE) + diff_EM - die_M
        E_m[i] += E_m[i - 1] + div_E + (1 - Na_div_flag) * diff_NaE - (die_E + diff_EM)

        # Evaluate sums and create masks
        sum_e_m = np.sum(E_m[i])
        e_m_nonzero_mask = E_m[i] > 0
        n_m_nonzero_mask = N_m[i] > 0
        na_m_nonzero_mask = Na_m[i] > 0
        # bound_na_sum_nonzero_mask = Na_m[i] > 0 | bound_N
        ma_m_nonzero_mask = Ma_m[i] + Ma_m[i-1] > 0
        n_na_sum_nonzero_mask = n_m_nonzero_mask | na_m_nonzero_mask

        # store time cells spend in effector effector
        T_E += dt * e_m_nonzero_mask if sum_e_m > 0 else 0.0
        die_M = 0*die_M

        # store time an effector spends in cytotoxic state
        for_where = ((1 - Na_div_flag)*(Na_m[i-1] - diff_NaE) + div_Ma + diff_EM + ma_m_nonzero_mask*memory_death > 0)
        where_output = np.where(for_where)[0]
        for l in where_output:

            if memory_death:
                if Ma_m[i,l] > 0:
                    T_Ecyt[l] = list(itertools.compress(T_Ecyt[l], survive_M[l])) + ((1 - Na_div_flag[l])*(Na_m[i-1,l] - diff_NaE[l]) + div_Ma[l]) * [0.0] + diff_EM[l] * [T_E[l].item()]
                    arr = np.array(T_Ecyt[l])
                    survive_M[l] = (rng.random(size = Ma_m[i,l]) > (arr > 0)*dt/memory_lifespan(arr, min_time = dt))
                    die_M[l] = Ma_m[i,l] - np.sum(survive_M[l])
                else:
                    T_Ecyt[l].extend(((1 - Na_div_flag[l])*(Na_m[i-1,l] - diff_NaE[l]) + div_Ma[l]) * [0.0] + diff_EM[l] * [T_E[l].item()])
                    arr = np.array([])
                    survive_M[l] = []
                    die_M[l] = 0
            else:
                T_Ecyt[l].extend(((1 - Na_div_flag[l])*(Na_m[i-1,l] - diff_NaE[l]) + div_Ma[l]) * [0.0] + diff_EM[l] * [T_E[l].item()])

        # Update division flag to allow division to proceed
        div_count += div_Na + div_E + div_Ma
        div_E_count += div_E
        div_M_count += div_Ma
        Na_div_flag = (Na_m[i] < max_Na)*(na_m_nonzero_mask)

        #### New binding events ####
        r_N_bind = (n_m_nonzero_mask) * (1 - bound_N) * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = 0.0, psi_2 = 1, psi_3 = 0.0, L_0 = 0, 
            K_1 = S_max, K_2 = K_H, K_3 = K_H # The rate of non-specific binding is important
        ) / (char_times[0]) if cells_sensed >= 1 else 0.0

        #### Transition rates modulated by antigen and cytokine signals ####
        r_Na_div = 1 / char_times[2]
        
        r_Ma_div = 0 * (div_count < max_expand) / char_times[4]
        
        r_Na[i] += bound_N*f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = psi_myc_I, psi_2 = psi_myc_HI, psi_3 = psi_myc_HE, L_0 = L0_Na,
            K_1 = K_I,
            K_2 = K_H, K_3 = K_H
        )/char_times[6] * n_m_nonzero_mask

        r_NaE[i] += f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = psi_NE_I, psi_2 = psi_NE_HI, psi_3 = psi_NE_HE,L_0 = L0_NE,
            K_1 = K_I,
            K_2 = K_H, K_3 = K_H)/char_times[6] * na_m_nonzero_mask

        r_E_div[i] += (div_count < max_expand) * f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = psi_myc_I, psi_2 = psi_myc_HI, psi_3 = psi_myc_HE, L_0 = L0_Na,
            K_1 = K_I,
            K_2 = K_H, K_3 = K_H
        )/char_times[6] * e_m_nonzero_mask
        
        r_EM[i] += f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = psi_EM_I, psi_2 = psi_EM_HI, psi_3 = psi_EM_HE, L_0 = L0_EM,
            K_1 = K_I,
            K_2 = K_H, K_3 = K_H)/char_times[6] * e_m_nonzero_mask

        r_Edie[i] += f_XtoY(
            sig_1 = p_tcr*cells_sensed, sig_2 = p_cyt*HI[i], sig_3 = p_cyt*HE[i],
            psi_1 = psi_Edie_I, psi_2 = psi_Edie_HI, psi_3 = psi_Edie_HE, L_0 = L0_Edie,
            K_1 = K_I,
            K_2 = K_H, K_3 = K_H)/char_times[6] * e_m_nonzero_mask

        cum_r_NaE += r_NaE[i]

        #### Store differentiation biases
        to_add_to_bias = np.zeros(4)

        if np.sum(n_m_nonzero_mask) > 0.:
            to_add_to_bias[0] = np.mean(r_Na[i][n_m_nonzero_mask])
        
        if np.sum(na_m_nonzero_mask) > 0.:
            to_add_to_bias[1] = np.mean(r_NaE[i][na_m_nonzero_mask])
            
        if sum_e_m > 0.:
            to_add_to_bias[2] = np.mean(r_E_div[i][e_m_nonzero_mask])
            to_add_to_bias[3] = np.mean(r_Edie[i][e_m_nonzero_mask])
            
        bias_t[i] += to_add_to_bias

        # Increment time
        t += dt

    # Collect population dynamics
    N, Na, Ma, E = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(Ma_m, axis = 1), np.sum(E_m, axis = 1)

    T_pEcytM = np.concatenate(T_Ecyt) if N_0 > 0 else np.array([0.0]) # time that memory spends in effector
    count_cM = np.sum(T_pEcytM == 0
                     ) if len(T_pEcytM[T_pEcytM == 0]) > 0 and N_0 > 0.0 else 0.0
    count_eM = np.sum(T_pEcytM > 0
                     ) if len(T_pEcytM[T_pEcytM == 0]) > 0 and N_0 > 0.0 else 0.0

    # Project out surviving primary memory
    mean_frac_cM = count_cM/np.sum(np.amax(Na_m, axis = 0)) if np.sum(np.amax(Na_m, axis = 0)) > 0.0 else 0.0
    mean_T_pEcyteM = np.mean(T_pEcytM[T_pEcytM > 0]) if len(T_pEcytM[T_pEcytM > 0]) > 0 and N_0 > 0.0 else 0.0

    zeros_n_0_var = np.zeros(N_0_var)

    selection_ratio = np.sum(div_E_count*p_tcr)/np.sum(div_E_count) if np.sum(div_E_count) > 0 else 0.0

    lineage_comp = np.vstack([
        np.amax(Na_m, axis = 0), div_M_count, div_E_count,
        p_tcr + zeros_n_0_var, p_cyt + zeros_n_0_var
    ])

    dyn_data = np.array([S, I, N, Na, E, Ma, HI, I_d_I + I_d_E*(infection_model != 'cancer'), S_d_E, HE]).T # add I_d_E as a separate variable
    prim_bias = bias_t

    ts = np.linspace(0, duration, int_steps + 1)

    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, pI, N, Na, pE, pM, pHI, pI_d_I, pI_d_SE, pHE = dyn_data[:,0], dyn_data[:,1], dyn_data[:,2], dyn_data[:,3], dyn_data[:,4], dyn_data[:,5], dyn_data[:,6], dyn_data[:,7], dyn_data[:,8], dyn_data[:,9]

    dt = ts[1] - ts[0]

    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, K_I,
              d_H, K_H, K_S,
              N_0, max_Na]),
              char_times,
              activation_regulation, NE_regulation, EM_regulation, contraction_regulation))

    sim_summary = np.array([I[-1]/I_0,
                       np.argmax(pI)*dt,
                       np.argmax(pI < I_min)*dt,
                       np.amax(pI_d_I) + 0*I[-1],
                       np.amax(pI_d_SE),
                       np.amax(pE),
                       np.argmax(pE)*dt*(np.max(pE + pM) >= max_Na) if N_0 > 0 else duration,
                       (np.argmax((pE + pM) >= N_0, axis = 0)*dt + duration*(np.amax(pE + pM) < N_0)) if N_0 > 0 else duration,
                       count_eM,
                       mean_T_pEcyteM,
                       (duration - (np.argmax(pE[::-1] >= N_0 , axis = 0)*dt))*(np.max(pE) >= N_0) if N_0 > 0 else 0.0,
                       mean_frac_cM,
                       np.amax(pHE),
                       np.amax(pHI),
                       pE[-1]])

    # down-size timeseries
    pnts = int(0.1*steps)
    keep = [i*10 for i in np.arange(0,pnts)]

    if out_data == "full":
        out_dict = {"reg_coeffs": np.concatenate((activation_regulation, NE_regulation, EM_regulation, contraction_regulation)), 
                    "cell_time_series": dyn_data, 
                    "time": ts, 
                    "lineage_diff": lineage_comp, 
                    "prim_diff_bias": prim_bias, 
                    "eff_by_lin": (E_m), 
                    "parameters": parameters, 
                    "summary_stats": sim_summary, 
                    "pmemory_formed": T_pEcytM if N_0 > 0 else []}
    elif out_data == "small":
        out_dict = {"cell_time_series": dyn_data[keep], 
                    "prim_diff_bias": prim_bias[keep],
                    "parameters": parameters, 
                    "summary_stats": sim_summary}

    return out_dict


stat_names = [r"$I_{p}(T_{sim})$",
              r"$T_{I_p}^{max}$",
              r"$T_{I_p}^{min}$",
              r"$\int_0^{T_{sim}} (d_I+d_{I,E})I_{p}dt$",
              r"$\int_0^{T_{sim}} d_{I,S}\cdot S_{p}dt$",
              r"$E_p^{max}$",
              r"$T_{E_p}^{max}$",
              r"$T_{E_p}^{start}$",
              r"$(eM)^{max}$",
              r"$\langle T_{E} \rangle_{eM}$",
              r"$T_{E_p}^{end}$",
              r"$cM/(E + cM)$",
              r"$HE_p^{max}$",
              r"$HI_p^{max}$",
              r"$E(T_{sim})$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$d_H$", r"$K_{E,I}$", r"$K_{E,H}$", r"$K_{E,S}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N,A_{in}}$", r"$\tau_{N \cdot A_{in}}$", r"$\tau_{N^*,N^*}$", r"$\tau_{E_,E}$", r"$\tau_{M,M}$", r"$\tau_{E_{die}}$", r"$\tau_{N^*}$",
               r"$\psi_{N^*}^{Ag}$", r"$\psi_{N^*}^{H_I}$", r"$\psi_{N^*}^{H_E}$", r"$\ell_{N \to N^*}$",
               r"$\psi_{N,E}^{Ag}$", r"$\psi_{N,E}^{H_I}$", r"$\psi_{N,E}^{H_E}$", r"$\ell_{N^{*} \to E}$",
               r"$\psi_{E,M}^{Ag}$", r"$\psi_{E,M}^{H_I}$", r"$\psi_{E,M}^{H_E}$", r"$\ell_{E \to M}$",
               r"$\psi_{E,\emptyset}^{Ag}$", r"$\psi_{E,\emptyset}^{H_I}$", r"$\psi_{E,\emptyset}^{H_E}$", r"$\ell_{E \to \emptyset}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'K_I',
                      'd_H', 'K_H', 'K_S',
                      'N_0', 'max_Na',
                      't_bind', 't_unbind', 't_Na_div', 't_E_div', 't_M_div', 't_E_die', 't_cycle',
                      'psi_myc_I', 'psi_myc_HI', 'psi_myc_HE', 'L0_Na',
                      'psi_NE_I', 'psi_NE_HI', 'psi_NE_HE', 'L0_NE',
                      'psi_EM_I', 'psi_EM_HI', 'psi_EM_HE', 'L0_EM',
                      'psi_Edie_I', 'psi_Edie_HI', 'psi_Edie_HE', 'L0_Edie']

stat_names_for_df = ['p_load', 'T_max_pI', 'T_min_pI', 'harm_pI',
                     'harm_pS', 'max_pE', 'T_pE_max', 'T_pE_start',
                     'max_eM', 'T_pEcyteM', 'T_pE_end', 'frac_cM',
                     'int_pHE', 'int_pHI', 'E_end']

Na_reg = ['psi_myc_I', 'psi_myc_HI', 'psi_myc_HE', 'L0_Na']
NE_reg = ['psi_NE_I', 'psi_NE_HI', 'psi_NE_HE', 'L0_NE']
EM_reg = ['psi_EM_I', 'psi_EM_HI', 'psi_EM_HE', 'L0_EM']
EE_reg = ['psi_Edie_I', 'psi_Edie_HI', 'psi_Edie_HE', 'L0_Edie']

module_labels = ['$N \longrightarrow N^*$', '$N^* \longrightarrow E$', '$E \longrightarrow \emptyset$']
modules = ['Na', 'NE', 'EE']
reg_stim = Na_reg[0:3] + NE_reg[0:3] + EE_reg[0:3]
reg_bl = [Na_reg[3]] + [NE_reg[3]] + [EE_reg[3]]
reg_bl_label = [param_names[-13]+'\n'+r", $\ell_{E \to 2E}$"] + [param_names[-9]] + [param_names[-1]]

perf_vars = ["peff_total_harm", "peff_infection", "peff_toxicity", "max_pE"] #, "max_eM_fold", "log_T_pEcyteM"] #, "max_pE_fold", "T_pE_start", 'T_pE_clear']
perf_labels = ["total harm", "infection harm", "toxicity", "max effector"] #, "E. memory\n expansion [$N_0$]", "Time as\n effector [day]"] #, "Effector\n expansion [$N_0$]", "Resp. timing \n [day]", "Resp. clear \n [days]"]

key_var = 'antigenicity_over_harm'
key_var_label = "Ag.-inflam. salience\n"+r"$\frac{\tau_I^{-1}}{\tau_H^{-1}}$"

vir_vars = ['d_I', 'K_I', 'b_I', 'K_H', 'N_0', 'S_0', 'I_0']

reg_markers = ['o', 's', 'v', '^', 'X','*', 'D','>', '<']
signal_classes = ["Ag. sens.", "Inflam. sens.", "Anti-inflam. sens.", "Baseline"]
opt_vars_labels = [r"$\psi_\text{Ag}$",
                   r"$\psi_\text{inf}$",
                   r"$\psi_\text{res}$",
                   r"$g_{0,N \to N^*}=g_{0,E \to 2E}$",
                   r"$g_{0,N^* \to E}$",
                   r"$g_{0,E \to \emptyset}$"]