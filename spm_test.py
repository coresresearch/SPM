# LiC6 -> C6 + Li+ + e- (reaction at the anode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential,Species,Participant,Half_Cell
 
'''
USER INPUTS
'''
# 
i_ext = -2000 # external current [A/m^2]
Phi_dl_0 = 1.8 #-U+ 0.75 # initial value for Phi_dl [V]
X_Li_0 = 0.5 # Fraction of the Anode that is ocupied by Lithium [-]
MV = 13*10**-6 # Molar Volume of Lithium [m^3/mol]
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
#C_Li_plus = 1 # Concentration of Li+ in the Electrolyte [mol/L] (for use later on when U is not constant)
i_o = 120 # Exchange Current Density [A/m^2] 
C_dl = 6*10**-2 # Double Layer Capacitance [F/m^2]
T = 273.15 + 25 # standard temperature [K]
U = 1 # Open circuit (equalibrium) potential [V]
n = 1 # number of electrons [mol_electrons/mol_rxn]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
Delta_y = 50*10**-6 # Anode thickness [m]
r = 5*10**-6 # particle radius [m]
Epsilon_g = 0.65 # volume fraction fo graphite [-]

'''
Constants and Parameters
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]
BnF_RT_a = Beta*n*F/R/T # combine constants for simplicity
BnF_RT_c = (1-Beta)*n*F/R/T

# Geometry
V = (4*np.pi/3)*r**3 # Volume of the anode [m^3]
A_surf = 4*np.pi*r**2 # surface area of the anode [m^2]
A_s = 3/r # ratio of graphite surface area to volume [1/m]
A_sg = Epsilon_g*Delta_y*A_s # surface area per geometric area [m^2_interface/m^2_geometric]

nuA_nF = nu_Li_plus*A_s/n/F # combine constants for convenience

# Initial Values
C_Li_0 = X_Li_0/MV # initial value for Lithium in the Anode [mol_Li/m^3]
#print(N_Li_0)

## Derivation 
# Eta_an = Phi_an - Phi_el - U ; Phi_an = 0 ; Phi_an = Phi_el - delta_Phi_dl_an
# Eta_an = - delta_Phi_dl_an - U => sub into Butler Volmer
# i_ext = i_dl + i_far ; -i_dl = C_dl*(d Delta_Phi_dl/ dt)
# (i_far - i_ext)/C_dl = d Delta_Phi_dl/dt
def residual(_,SV,i_ext,U,BnF_RT_a,BnF_RT_c,C,i_o,A_surf,nuA_nF):
    dSVdt = np.zeros_like(SV)
    V = SV[0]
    i_far= faradaic_current(i_o,V,U,BnF_RT_a,BnF_RT_c)
    
    dPhi_dl_dt = (i_far - i_ext/A_sg)/C  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    dC_Li_dt = i_far*nuA_nF
    return [dPhi_dl_dt,dC_Li_dt]

# Integration
t_start = 0 # [s]
t_end = .001 # length of time passed in the integration [s]
t_span = [t_start,t_end]
SV_0 = [Phi_dl_0,C_Li_0] # initial values
N = 10000 # number of time steps
dt = t_end/(N-1) # length of each time step [s]

# I could not get this to work, it ket saying not enought input arguments 
#SV = residual(set_up_ivp,t_span,SV_0,method = 'BDF',args=[(i_ext,U,BnF_RT,C_dl,i_o,A_surf,nuA_nF),]) 
SV = solve_ivp(residual,t_span,SV_0,method='BDF',args=(i_ext,U,BnF_RT_a,BnF_RT_c,C_dl,i_o,A_surf,nuA_nF),
               rtol = 1e-5,atol = 1e-8)
    
   # lambda t, delta_Phi_dl: residual(t,delta_Phi_dl,i_ext,U,BnF_RT_a,BnF_RT_c,C_dl,i_o,A_surf,nuA_nF),t_span,SV_0,t_eval=np.linspace(t_start,t_end,N))

## Post Processing
# DL Potential Difference 
plt1 = plt.figure(1)
plt.plot(SV.t,SV.y[0])
plt.xlabel("time [s]")
plt.ylabel("Change in Potential Across the Double Layer [V]")
plt1.show()

# Current
plt2 = plt.figure(2)
i_fa = np.zeros_like(SV.t)
for ind, ele in enumerate(i_fa):
    i_fa[ind] = faradaic_current(i_o,SV.y[0,ind],U,BnF_RT_a,BnF_RT_c)
i_dl = i_ext/A_sg - i_fa
plt.plot(SV.t,i_dl,'o')
plt.plot(SV.t,i_ext/A_sg*np.ones(len(i_dl)))
plt.plot(SV.t,i_fa)
plt.xlabel("time [s]")
plt.ylabel("Current [A]")
plt.ylim((-250,250))
plt.legend(['DL','EXT','FAR'])
plt.show()

'''
# Concetration
plt3 = plt.figure(3)
plt.plot(SV.t,SV.y[1])
plt.xlabel("time [s]")
plt.ylabel("Amount of Li in the Anode [mol]")

plt3.show()
'''


