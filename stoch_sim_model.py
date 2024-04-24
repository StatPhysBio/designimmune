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
d_S = 0.05
I_0 = S_0/10**5 # initial detectable levelof infected cells
b_I = 10**(-6) # harm per unit virion (Chao et al. 2004, Iwami et al. 2015)
d_IE = 12 # effector clearance rate of infection: 2-16 day^(-1) Halle et al. (2016)
K_IE_min = 10**4
K_IE = 10**4 # effector avidity (half-max) for infected cells at low infection concetrations (Mayer et al 2019; Chao et al. 2004)
K_EI = K_IE
d_I = np.minimum(5*d_S, S_0*b_I) # successful virus cannot kill cells faster than it infects new ones

# APC dynamics
Aout_0 = 10**4
b_Aout = 2.0
b_Ain = 0.5

# Inflammatory response
d_H = 2.0
b_H = 10 # innate response/inflammation per lysed cell compared to natural death
l_H = 1 # cooperativity
K_IH = 0.01*d_S*S_0 # half-max level of instantaneous damage required to trigger innate/inflammatory response
K_EH = 1*K_IH # half-max level of inflammation required to trigger lymphocyte response
ep = 10**(-3) # off-target rate of harm
K_SE = 10**8
kappa = 0.5 # maximal reduction in replication rate due to inflammatory response
d_IH = d_IE*ep
H_0 = 0.0

# Immune cells
N_0 = 300
max_Na = 2**3
max_expand = 2**17 #(Marchingo et al.)
t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_M_diff, t_E_die, t_E_cyt = 1, 3/4, 1/4, 1/3, 1/2, 20.0, 2.5, 2/3
rel_persist_M = 5 # d_eM/d_cM

# Division timer
d_myc = np.log(2)*24/7 # *np.log(2)
myc_thresh = 10**(2.6)
b_myc = 2*myc_thresh/t_bind

# hyper parameters
alpha = 0.5 # weight of antigenic signals relative to inflamatory signals
vir_prop = np.vstack((np.array([0, K_SE, 1.0, 1.0]), # autoimmune situation
                      np.array([0, K_SE, 0.25, 1.0]),
                      np.array([0, K_SE, 1.0, 0.25]),
                      np.array([0, K_SE, 0.25, 0.25]),
                      np.array(np.meshgrid(d_S*np.array([2, 5]), # vary d_I
                                            K_IE*np.array([1, 10]), # vary K_IE
                                            np.array([0.25,1.0]), # vary K_EI
                                            np.array([0.25,1.0]))).T.reshape(-1,4))) # vary K_EH

# define reg options
psi_max = 2.0
xv, yv = np.meshgrid(np.linspace(-1, 1, 31), np.linspace(-1, 1, 31))
psi_2d = psi_max*np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],[np.sin(np.pi/4), np.cos(np.pi/4)]]).dot(np.vstack([xv.ravel(), yv.ravel()]))/np.sqrt(2)
psi_opts = np.vstack((np.array([0,0,0]),
                      np.vstack([np.array([[x[0], x[1], psi_max - np.abs(x[0]) - np.abs(x[1])], [x[0], x[1], -(psi_max - np.abs(x[0]) - np.abs(x[1]))]]) for x in psi_2d.T])))
psis = psi_max*np.array([-0.5, -0.5, 0.0]) # regulatory weights: psi_M_I, psi_M_H, psi_M_IH
F_0s = np.array([0.0, 0.0])

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


def f_XtoY(I_sig, H_sig, psi_I, psi_H, psi_IH, F_0, K_I, K_H, reg_model = "mwc_like"):
    ### variable
    # I_sig := antigenic stimuli
    # H_sig := inflammatory stimuli
    # psi_. := strength of regulatory action (psi > 0 means upregulation, psi = 0 means no regulation, and psi < 0 means down regulation
    # F_0 := bias towards transitioning from state X to Y in the absense of stimuli
    # K_. := associated concentration thresholds for the tranistions
    # reg_model := family of regulatory functions considered: Monod-Wyman-Changeaux inspired, and Hill functions
    
    if reg_model == "mwc_like":
        F_1 = psi_I*np.log((1 + I_sig/K_I)/2) + psi_H*np.log((1 + H_sig/K_H)/2) + psi_IH*np.log((1 + (I_sig*H_sig)/(K_I*K_H))/2)
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
                    Aout_0 = Aout_0, b_Ain = b_Ain, b_H = b_H, d_H = d_H, K_EI = K_EI, K_EH = K_EH, kappa = kappa,
                    N_0 = N_0, max_Na = max_Na, b_myc = b_myc, d_myc = d_myc, myc_thresh = myc_thresh, max_expand = max_expand,
                    char_times = [t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_M_diff, t_E_die, t_E_cyt],
                    regulation_coeffs = psis,
                    regulation_bias = F_0s,
                    alpha = alpha,
                    infection = "prim",
                    vir_model = "indep_harm",
                    duration = sim_duration, 
                    steps = sim_steps,
                    out_data = "small"):
    
    # VARIABLE DEFINITIONS:
    # S_0 := S_0, I_0 := I_0, b_S := b_S, b_I := b_I, d_S := d_S, d_I := d_I, d_IE := d_IE, d_IH := d_IH, K_IE := K_IE, K_IH := K_IH,
    # Aout_0 := Aout_0, b_Ain := b_Ain, b_H := b_H, d_H := d_H, K_EI := K_EI, K_EH := K_EH,
    # N_0 := N_0, max_Na := max_Na, b_myc := b_myc, d_myc := d_myc, myc_thresh := myc_thresh,
    # char_times := [t_act, t_bind, t_Na_div, t_E_div, t_M_div, t_M_diff, t_E_die],
    
    #################################
    ### SET META-VARIABLES FOR SIMULATION ###
    #################################
    dt =  duration/steps
    N_0_var = int(N_0)
    b_S = S_0*d_S
    
    # in case of autoimmune response
    if d_I == 0.0:
        d_Sauto = 0.01*d_S 
        K_SE = K_SE/1000
        K_EI = K_SE
        
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
    
    # select virulence model:
    if vir_model == "dep_harm": # makes it so that the average virus produced is roughly the same independent of infected eath rate but low
        b_I = b_I + d_I/S_0
    
    # draw population of reponding cells for agent-based simulations
    psi_M_I, psi_M_H, psi_M_IH = regulation_coeffs
    F0_NM, F0_EM = -np.sum(regulation_coeffs), -np.sum(regulation_coeffs)
    
    mu_tcr = 0.8
    var_tcr = mu_tcr*(1-mu_tcr)*0.9
    mu_cyt = 0.8
    var_cyt = mu_cyt*(1-mu_cyt)*0.9
    
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
            N_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
            N_m[0,:] +=1
        elif k == 1: # secondary infection
            N_m[0,:] = 0*N_m[-1,:] # no naive cells during secondary infection
            M_survive = np.random.binomial( np.ones(len(T_EcytM), dtype =int), np.exp(-1/rel_persist_M - (1 -1/rel_persist_M)*T_EcytM/char_times[6]) )
            M_count = int(np.sum(M_survive))
            T_EcytM = T_EcytM[M_survive > 0]
            M_m = np.zeros((int(steps)+1, M_count), dtype = int)
            M_m[0,:] += 1
            act_M = np.zeros(M_count, dtype =int)

        div_E_count = np.zeros(N_0_var if k == 0 else M_count)
        diff_EM_count = np.zeros(N_0_var if k == 0 else M_count, dtype =int)
        
        Na_m = np.zeros((int(steps)+1, N_0_var), dtype = int)
        Ma_m = np.zeros((int(steps)+1, N_0_var if k == 0 else M_count), dtype = int)
        E_m = np.zeros((int(steps)+1, N_0_var if k == 0 else M_count), dtype = int) # effector in periphary
        T_E = np.zeros(N_0_var if k == 0 else M_count)
        T_Ecyt = np.array(N_0_var*[''] if k == 0 else M_count*[''])
        
        mycNa_m = np.zeros((int(steps)+1, N_0_var))
        mycMa_m = np.zeros((int(steps)+1, N_0_var if k == 0 else M_count))
        mycE_m = np.zeros((int(steps)+1, N_0_var if k == 0 else M_count))
        if k == 1:
            mycM_m = np.zeros((int(steps)+1, M_count))
        
        bias_t = np.zeros((int(steps)+1, 3))
        
        # Define event timer variables
        unbound_Na = np.zeros(N_0_var, dtype =int)
        Na_div_flag = np.ones(N_0_var, dtype =int)
        
        mycNa = np.zeros(N_0_var)
        mycE = np.zeros(N_0_var if k == 0 else M_count)
        mycMa = np.zeros(N_0_var if k == 0 else M_count)
        if k == 1:
            mycM = 2*myc_thresh*np.ones(M_count)
        
        p_NaM = np.zeros((int(steps)+1, N_0_var))
        p_EM = np.zeros((int(steps)+1, N_0_var if k == 0 else M_count))
        if k == 1:
            p_MM = np.zeros((int(steps)+1, M_count))
        
        b_unbind_t = np.zeros(N_0_var)
        b_N_act = np.zeros(N_0_var)
        b_Na_div = np.ones(N_0_var)/char_times[2]
        b_E_div = np.ones(N_0_var if k == 0 else M_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
        b_Ma_div = np.ones(N_0_var if k == 0 else M_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]
        d_E_die = np.ones(N_0_var if k == 0 else M_count)/char_times[6]
        if k == 1:
            b_M_act = np.zeros(M_count)
            b_M_div = np.ones(M_count)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[2]   
        
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
            Na_pop, E_pop, Ma_pop = np.sum(Na_m[i-1]), np.sum(E_m[i-1]), np.sum(Ma_m[i-1])
            
            if k == 1:
                M_pop = np.sum(M_m[i-1])
                
            #### Run infection dynamics: replication and effector clearance ####
            
            # Update state of susceptible, infected, APCs, and inflammation
            S[i] = S[i-1] + dt*(b_S - (d_S + d_Sauto*np.exp(-((t-1.0)/t_Hauto)**2))*S[i-1] - (I[i-1] >= I_0)*b_I*I[i-1]*S[i-1]*(1-kappa*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H)) - S[i-1]*(d_IH*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*(E_pop)/(K_SE + S[i-1] + E_pop)))*(S[i-1] >= 0.0)
            
            I[i] = I[i-1] + dt*((I[i-1] >= I_0)*b_I*I[i-1]*S[i-1]*(1-kappa*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H)) - d_IH*I[i-1]*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) - d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop) - (d_I)*I[i-1])*(I[i-1] >= 0.0)
            
            cell_lysed = (d_I)*I[i-1] + 0*d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop) + S[i-1]*(d_Sauto*np.exp(-((t-1.0)/t_Hauto)**2) + 0*d_IE*(E_pop)/(K_SE + S[i-1] + E_pop))
            
            H[i] = H[i-1] + dt*(b_H*(cell_lysed) - d_H*H[i-1])*(H[i-1] >= 0.0)
            
            Aout[i] = Aout[i-1] + dt*(b_Aout*Ain[i-1] - b_Ain*Aout[i-1]*H[i-1]**l_H/((K_EH)**l_H + H[i-1]**l_H))*(Aout[i-1] >= 0.0)
            
            Ain[i] = Ain[i-1] + dt*(b_Ain*Aout[i-1]*H[i-1]**l_H/((K_EH)**l_H + H[i-1]**l_H) - b_Aout*Ain[i-1] - d_IE*Ain[i-1]*Na_pop/(K_IE + Na_pop))*(Ain[i-1] >= 0.0)
            
            I_d_I[i] = I_d_I[i-1] + dt*(I[i-1] >= I_0)*d_I*I[i-1] + (I[i] if i == int(steps) else 0) # cells killed by infection
            
            I_d_IE[i] = I_d_IE[i-1] + dt*(I[i-1] >= I_0)*(d_IH*I[i-1]*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*I[i-1]*(E_pop)/(K_IE + I[i-1] + E_pop)) # cells killed by immune response
            
            I_d_S[i] = I_d_S[i-1] + dt*S[i-1]*(d_IH*H[i-1]**l_H/(K_IH**l_H + H[i-1]**l_H) + d_IE*(E_pop)/(K_SE + S[i-1] + E_pop) + d_Sauto*np.exp(-((t-1.0)/t_Hauto)**2))
            
            ## I. Recruitment/Priming

            # (a) Phase 1: Naive cells encounter and bind APCs
            act_N = np.random.binomial(N_m[i-1], b_N_act*dt)

            # (a) Phase 2: Activated naive cells are bound to APCs and receive stimulation.

            # (a) Phase 3: Unbound activated naive cells divide and then differentiate
            div_Na = np.random.binomial(Na_m[i-1]*unbound_Na*Na_div_flag, dt*b_Na_div)

            diff_NaM = np.random.binomial(Na_m[i-1]*(1-Na_div_flag), np.sum(p_NaM[0:i-1], axis = 0))
            
            # (b) Memory cells from a prior infection activate quickly and divide
            if k == 1:
                act_M += np.random.binomial(M_m[i-1] - act_M, b_M_act*dt)
                div_M = np.random.binomial(act_M, dt*b_M_div)
                diff_MM = np.random.binomial(2*div_M, np.sum(p_MM[np.maximum(0,i-1-int(char_times[2]/dt)):i-1], axis = 0))
            
            ## II. Expansion

            # (a) New central memory cells divide
            div_Ma = np.random.binomial(Ma_m[i-1] - diff_EM_count, dt*b_Ma_div)

            # (b) Effector cells divide, differentiate, die
            die_E = np.random.binomial(E_m[i-1], d_E_die*dt)
            div_E = np.random.binomial(E_m[i-1] - die_E, dt*(b_E_div))
            diff_EM = np.random.binomial(E_m[i-1] - die_E + div_E, p_EM[i-1])
            
            #### Update population dynamics: ####
            N_m[i] = N_m[i-1] - act_N
            Na_m[i] = Na_m[i-1] + div_Na + act_N - (1 - Na_div_flag)*Na_m[i-1]
            
            if k == 1:
                M_m[i] = M_m[i-1] - div_M
                act_M += -div_M
                
            Ma_m[i] = Ma_m[i-1] + div_Ma + ((1 - Na_div_flag)*(diff_NaM) if k == 0 else 0) + (diff_MM if k == 1 else 0) + diff_EM
            E_m[i] = E_m[i-1] + div_E + ((1 - Na_div_flag)*(Na_m[i-1] - diff_NaM) if k == 0 else (2*div_M - diff_MM)) - (die_E + diff_EM)
            
            # Update division flag to allow differentiation to proceed
            Na_div_flag = 1*(Na_m[i] < max_Na)
            div_E_count += (div_Na if k == 0 else 0) + (div_M if k == 1 else 0) + div_E + div_Ma
            diff_EM_count += diff_EM
            
            #### New binding events ####
            b_N_act = p_tcr*Ain[i]/(char_times[0]*(N_0 + Ain[i]))*(Ain[i] >= 1)
            b_unbind_t = np.fmin(2*(Na_m[i] == 1)*(i - np.argmin(N_m[0:i], axis = 0))*dt/(f_XtoY(1/K_IE, H[i], psi_I = psi_max/4, psi_H = psi_max/4, psi_IH = psi_max/2, F_0 = 0.0, K_I = 1/(10*K_IE_min), K_H = K_EH)*char_times[1])**2, 1/dt) if np.sum(Na_m[i]) >= 1 else 0.0
            # np.exp(-np.log10(np.sqrt(2))*np.log(K_IE/K_IE_min))
            unbound_Na += np.random.binomial(1-unbound_Na, b_unbind_t*dt)
            
            if k == 1:
                b_M_act = p_tcr*Ain[i]/(char_times[0]*(M_count + Ain[i]))*(Ain[i] >= 1)
            
            #### MYC Dynamics ####
            mycNa = (mycNa + dt*(b_myc*(1-unbound_Na)))*(Na_m[i] > 0)
            # (1-H[i]**l_H/((K_EH/p_cyt)**l_H + H[i]**l_H))
            
            mycE = (mycE - dt*(f_XtoY(p_tcr*I[i], p_cyt*H[i], psi_I = 0.0, psi_H = -psi_max, psi_IH = 0.0, F_0 = psi_max, K_I = K_EI, K_H = K_EH)*mycE*d_myc))*(E_m[i] >= 1) + (mycNa*(Na_m[i] > 0) if k == 0 else mycM*(M_m[i] > 0))

            mycMa = (mycMa - dt*f_XtoY(p_tcr*I[i], p_cyt*H[i], psi_I = 0.0, psi_H = -psi_max, psi_IH = 0.0, F_0 = psi_max, K_I = K_EI, K_H = 2*K_EH)*(mycMa*d_myc))*(Ma_m[i] > 0) + (mycNa*(Na_m[i] >= 1) if k == 0 else mycM*(M_m[i] > 0)) # higher decay rate of myc
            
            if k == 1:
                mycM = (mycM + 0*dt*(b_myc*Ain[i]/(M_pop + Ain[i])))*(M_m[i] >= 1)

            #### transition probabilities modulated by antigen and cytokine signals ####
            p_NaM[i] = dt*f_XtoY(p_tcr*(1-unbound_Na), p_cyt*H[i], psi_M_I, psi_M_H, psi_M_IH, F_0 = F0_NM, K_I = 1/2, K_H = K_EH)*(1-unbound_Na)*(Na_m[i] == 1)/char_times[5] if k == 0 else 0.0
            p_EM[i] = dt*f_XtoY(p_tcr*(I[i] + S[i]*(d_I == 0.0)), p_cyt*H[i], psi_M_I, psi_M_H, psi_M_IH, F_0 = F0_EM, K_I = K_EI + np.sum(E_m[i]), K_H = K_EH)*(E_m[i] > 0)/char_times[5]
            
            if k == 1:
                p_MM[i] = dt*f_XtoY(p_tcr*(I[i] + S[i]*(d_I == 0.0)), p_cyt*H[i], psi_M_I, psi_M_H, psi_M_IH, F_0 = F0_NM, K_I = K_EI + np.sum(M_m[i]), K_H = K_EH)*(M_m[i] > 0)/char_times[5]

            #### Time-dependent rates modulated by antigen and cytokine signals ####
            b_E_div = (mycE > myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[3]
            b_Ma_div = (mycMa > myc_thresh)*(div_E_count < max_expand)*(alpha*p_tcr + (1-alpha)*p_cyt)/char_times[4]
            
            # store time cells become effector
            if k == 0:
                T_E += dt*((N_m[i] + Na_m[i]) > 0) if np.sum(Na_m[i]) > 0 else 0.0
                d_E_die = 2*(E_m[i] > 0)*np.maximum(0, i*dt - T_E)/char_times[6]**2
            elif k == 1:
                T_E += dt*(M_m[i] > 0) if np.sum(M_m[i]) > 0 else 0.0
                d_E_die = 2*(E_m[i] > 0)*np.maximum(0, T_EcytM + i*dt - T_E)/char_times[6]**2
               
            # store time an effector spends in cytotoxic state
            if k == 0 and np.sum(Na_m[i] + E_m[i]) > 0:
                T_Ecyt = np.char.add(T_Ecyt, 
                                     np.array([(',' if len(T_Ecyt[l])*(diff_NaM + diff_EM + div_Ma)[l] > 0 else '')+','.join( np.hstack((np.full(int( (diff_NaM + diff_EM)[l] ), 
                                                                                                    np.maximum(0, i*dt - T_E[l] ) if diff_EM[l] > 0 else 0.0, dtype='<U4'),
                                                                                                        np.zeros(int(div_Ma[l])))) )
                                               if (diff_NaM + diff_EM + div_Ma)[l] > 0 else '' for l in np.arange(0, N_0_var)]))

            #### Store myc levels ####
            mycNa_m[i] = mycNa
            mycMa_m[i] = mycMa
            mycE_m[i] = mycE
            
            if k == 1:
                mycM_m[i] = mycM

            #### Store differentiation biases
            bias_t[i] = np.array([np.mean((char_times[5]*p_NaM[i]/dt)[Na_m[i]*(1-unbound_Na) > 0]) if np.sum(Na_m[i]*(1-unbound_Na)) > 0.0 else 0.0, 
                                  np.mean((char_times[5]*p_EM[i]/dt)[E_m[i] > 0]) if np.sum(E_m[i]) > 0.0 else 0.0, 
                                  np.mean((char_times[5]*p_MM[i]/dt)[M_m[i] > 0]) if k == 1 else 0.0])
            
        # Increment time
            t += dt
        
        # Collect population dynamics
        N, Na, Ma, E = np.sum(N_m, axis = 1), np.sum(Na_m, axis = 1), np.sum(Ma_m, axis = 1), np.sum(E_m, axis = 1)
        if k == 1:
            M = np.sum(M_m, axis = 1)
        
        if k == 0:
            T_EcytM = np.hstack( ([np.fromstring(word, dtype = float, sep =',' ) for word in T_Ecyt]) ) # time that memory spends in effector
    
        lineage_comp = np.vstack([np.amax(N_m if k == 0 else M_m + Ma_m, axis = 0) ,
                                  np.amax(Ma_m if k == 0 else M_m + Ma_m, axis = 0),
                                  np.amax(E_m, axis = 0),
                                  p_tcr*np.ones(N_0_var if k == 0 else M_count),
                                  p_cyt*np.ones(N_0_var if k == 0 else M_count)])
        
        if k == 0: # primary infection
            dyn_data = np.array([S, I, Ain, Na, E, Ma, H, I_d_I + I_d_IE, I_d_S]).T
            prim_bias = bias_t #[p_NaM, p_EM]
        elif k == 1: # secondary infection
            dyn_data = np.hstack((dyn_data, np.array([S, I, Ain, Na + M, E, Ma, H, I_d_I+ I_d_IE, I_d_S]).T ))
            sec_bias = bias_t # [p_NaM, p_EM, p_MM]
                                 
    ts = np.linspace(0, duration, int(steps) + 1)
    
    # Compute summary statistics from simulations
    ## extract primary/secondary infection dynamics
    pS, sS, pI, sI, Ain, N, pE, sE, pM, sM, pH, sH, pI_d_I, sI_d_I, pI_d_S, sI_d_S = dyn_data[:,0], dyn_data[:,-9], dyn_data[:,1], dyn_data[:,-8], dyn_data[:,2], dyn_data[:, 3], dyn_data[:,4], dyn_data[:,-5], dyn_data[:,5], dyn_data[:,-4], dyn_data[:,6], dyn_data[:,-3], dyn_data[:,7], dyn_data[:,-2], dyn_data[:,8], dyn_data[:,-1]
        
    dt = ts[1]-ts[0]
    
    parameters = np.concatenate((np.array([S_0, I_0, b_I, d_S, d_I, d_IE, d_IH, K_IE, K_IH,
              Aout_0, b_Ain, b_H, d_H, K_EI, K_EH,
              N_0, max_Na, b_myc, d_myc, myc_thresh]),
              char_times,
              regulation_coeffs))

    sim_summary = np.array([np.sum(pI*dt), 
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
                       pM[-1],
                       sM[-1], 
                       np.sum(pE*dt), 
                       np.sum(sE*dt),
                       np.sum(pH*dt),
                       np.sum(sH*dt),
                       np.amin(pS),
                       np.amin(sS)])
    
    if out_data == "full":
        out_dict = {"reg_coeffs": np.array(regulation_coeffs), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "sec_diff_bias": sec_bias if k == 1 else [],"eff_by_lin": (E_m), "Na_myc_by_lin": mycM_m if k == 1 else mycNa_m, "Ma_myc_by_lin": mycMa_m, "E_myc_by_lin": mycE_m, "parameters": parameters, "summary_stats": sim_summary, "memory_persistence": T_EcytM}
    elif out_data == "small":
        out_dict = {"reg_coeffs": np.array(regulation_coeffs), "cell_time_series": dyn_data, "time": ts, "lineage_diff": lineage_comp, "prim_diff_bias": prim_bias, "sec_diff_bias": sec_bias if k == 1 else [], "parameters": parameters, "summary_stats": sim_summary}
    
    return out_dict


stat_names = [r"$\int_0^{T_{sim}} I_{p}dt$",
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
              r"$(M_p)^\infty$",
              r"$(M_s)^\infty$",
              r"$\int E_p dt$",
              r"$\int E_s dt$",
              r"$\int H_p dt$",
              r"$\int H_s dt$",
              r"$S_p^{min}$",
              r"$S_s^{min}$"]

param_names = [r"$S_0$",r"$I_0$", r"$b_I$", r"$d_S$", r"$d_I$", r"$d_{I,E}$", r"$d_{I,H}$", r"$K_{I,E}$", r"$K_{I,H}$",
               r"$A_{out}^{(0)}$", r"$b_{A_in}$", r"$b_H$", r"$d_H$", r"$K_{E,I}$", r"$K_{E,H}$",
               r"$N_0$", r"$N^*_{max}$", r"$b_D$", r"$d_D$", r"$D^*$",
               r"$\tau_{N^*,A_{in}}^{(+)}$", r"$\tau_{N^*,A_{in}}^{(-)}$", r"$\tau_{N^*}$", r"$\tau_E$", r"$\tau_{M}$", r"$\tau_{E_{die}}$",
               r"$\psi_{M}^{(I)}$", r"$\psi_{M}^{(H)}$", r"$\psi_{M}^{(I,H)}$"]


param_names_for_df = ['S_0', 'I_0', 'b_I', 'd_S', 'd_I', 'd_IE', 'd_IH', 'K_IE',
                      'K_IH', 'Aout_0', 'b_Ain', 'b_H', 'd_H', 'K_EI', 'K_EH',
                      'N_0', 'max_Na', 'b_myc', 'd_myc', 'myc_thresh',
                      't_act', 't_bind', 't_Na_div', 't_E_div', 't_M_div', 't_E_die',
                      'psi_M_I', 'psi_M_H', 'psi_M_IH']

stat_names_for_df = ['p_load', 's_load','T_max_pI', 'T_max_sI', 'harm_pI', 'harm_sI', 
                     'harm_pS', 'harm_sS', 'max_pE', 'max_sE','T_max_pE', 'T_max_sE', 
                     'inf_pM', 'inf_sM', 'int_pE', 'int_sE',
                     'int_pH', 'int_sH', 'min_pS', 'min_sS']