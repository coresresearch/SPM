# LiC6 -> C6 + Li+ + e- (reaction at the anode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Half_Cell_Eqlib_Potential,Butler_Volmer,Species,Participant,Half_Cell

# Held Constant
Vol_Frac = 0.5 # Fraction of the Anode that is ocupied by Lithium [m^3/m^3]
MV = 13*10**-6 # Molar Volume of Lithium [m^3/mol]
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
#C_Li_plus = 1 # Concentration of Li+ in the Electrolyte [mol/L] (for use later on when U is not constant)
i_o = 120 # Exchange Current Density [A/m^2] 
C_dl = 6*10**3 # Double Layer Capacitance [F/m^2]
T = 273.15 + 25 # standard temperature [K]
U = 1 # Open circuit (equalibrium) potential [V]
i_ext = -2 # external current [A]
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]
n = 1 # number of electrons [mol_electrons/mol_rxn]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
BnF_RT = Beta*n*F/R/T # combine constants for simplicity

# Geometry
r = 50*10**-6 # Anode radius [m]
V = (4*np.pi/3)*r**3 # Volume of the anode [m^3]
A_surf = 4*np.pi*r**2 # surface area of the anode [m^2]

nuA_nF = nu_Li_plus*A_surf/n/F # combine constants for convenience

# Initial Values
Phi_dl_0 = 0 #-U+ 0.75 # initial value for Phi_dl [V]
N_Li_0 = V*Vol_Frac/MV # initial value for Lithium in the Anode [mol_Li]
#print(N_Li_0)

## Derivation 
# Eta_an = Phi_an - Phi_el - U ; Phi_an = 0 ; Phi_an = Phi_el - delta_Phi_dl_an
# Eta_an = - delta_Phi_dl_an - U => sub into Butler Volmer
# i_ext = i_dl + i_far ; -i_dl = C_dl*(d Delta_Phi_dl/ dt)
# (i_far - i_ext)/C_dl = d Delta_Phi_dl/dt
def set_up_ivp(t,y,BnF_RT,C,i_o,A_surf,nuA_nF):
    i_far= i_o*A_surf*(math.exp(BnF_RT*(y[0]-U)) - math.exp(-BnF_RT*(y[0]-U)))
    
    dPhi_dl_dt = (i_far - i_ext)/C  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    dN_Li_dt = i_far*nuA_nF
    return [dPhi_dl_dt,dN_Li_dt]

# Integration
t_start = 0 # [s]
t_end = 1000 # length of time passed in the integration [s]
t_span = [t_start,t_end]
SV_0 = [Phi_dl_0,N_Li_0]#,N_Li_0] # initial values
N = 10000 # number of time steps
dt = t_end/(N-1) # length of each time step [s]

# I could not get this to work, it ket saying not enought input arguments 
#SV = solve_ivp(set_up_ivp,t_span,SV_0,method = 'BDF',args=[(BnF_RT,C_dl,i_o,A_surf,nuA_nF),]) 
SV = solve_ivp(lambda t, delta_Phi_dl: set_up_ivp(t,-delta_Phi_dl,BnF_RT,C_dl,i_o,A_surf,nuA_nF),t_span,SV_0,t_eval=np.linspace(t_start,t_end,N))

## Post Processing
# DL Potential Difference 
plt1 = plt.figure(1)
plt.plot(SV.t,SV.y[0])
plt.xlabel("time [s]")
plt.ylabel("Change in Potential Across the Double Layer [V]")
plt1.show()

# Current
plt2 = plt.figure(2)
i_dl = -C_dl*np.gradient(SV.y[0],dt)
i_fa = np.zeros(len(i_dl))
for ind, ele in enumerate(i_fa):
    i_fa[ind] = i_o*A_surf*(math.exp(BnF_RT*(-SV.y[0,ind]-U)) - math.exp(-BnF_RT*(-SV.y[0,ind]-U)))
plt.plot(SV.t,i_dl)
plt.plot(SV.t,i_ext*np.ones(len(i_dl)))
plt.plot(SV.t,i_fa)
plt.xlabel("time [s]")
plt.ylabel("Current [A]")
plt.ylim((-10,10))
plt.legend(['DL','EXT','FAR'])
plt2.show()

# Concetration
plt3 = plt.figure(3)
plt.plot(SV.t,SV.y[1])
plt.xlabel("time [s]")
plt.ylabel("Amount of Li in the Anode [mol]")

plt3.show()

input()

