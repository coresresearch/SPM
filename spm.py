# spm.py
#
# This file serves as the main model file.  
# It is called by the user to run the mode
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential,residual,Species,Participant,Half_Cell,Electrode

# LiC6 -> C6 + Li+ + e- (reaction at the anode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
# The rate Lithium ions enter the electrolyte from the anode is equal to the rate of Li+ enter the seperator and eventually the cathode
# I assume the concentration in the electrolyte is in uniform so there is no diffusion and there is no bulk movement of the fluid
#   so there is no convection therefore there is only migration which is driven by the potential difference
# Potential drop across the electroltye obeys Ohm's Law
# The cathode and anode have the same composition and dimensions 

# My main question are about how to apply the boundry conditions to make sure that the concentration of Li+ is the same
# flowing from the anode and into the cathode and the currents are the same throught. My current aproach feels disjointed
'''
USER INPUTS
'''
# 
i_ext = -2000 # external current [A/m^2]
Phi_dl_0 = 1.5 #-U+ 0.75 # initial value for Phi_dl [V]

nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
X_Li_0 = 0.5 # Initial Mole Fraction of the Anode for Lithium [-]
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
C_Li_plus = 1 # Concentration of Li+ in the Electrolyte [mol/L] (I assume electrolyte transport is fast)
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)

i_o = 120 # Exchange Current Density [A/m^2] 
Cap_dl = 6*10**-3 # Double Layer Capacitance [F/m^2]
sigma_s = 1.2 # Ionic conductivity for the seperator [1/m-ohm] (this is concentration dependent but it is constant for now)
t_s =  1e-5 # thickness of the seperator [m]
T = 298.15 # standard temperature [K]
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
# Do I need a Volume of the Electroltye, or can I get by with doing everything per unit area?
Vol_a = (4*np.pi/3)*r**3 # Volume of the anode [m^3]
A_surf = 4*np.pi*r**2 # geometric surface area of the anode [m^2]
A_s = 3/r # ratio of surface area to volume for the graphite anode [1/m]
A_sg = Epsilon_g*Delta_y*A_s # interface surface area per geometric surface area [m^2_interface/m^2_geometric]

nuA_nF = nu_Li_plus*A_s/n/F # combine constants for convenience

# Initial Values
X_g_0 = 1 - X_Li_0 # Initial Mole Fraction of Graphite [-]
N_g = Vol_a*rho_g/MW_g # The number of moles of graphite in the Anode [mol] (this number does not change)
C_g = N_g/Vol_a# Concentration of graphite in the anode [mol/m^3] (this number does not change)
N_Li_0 = N_g*(1/X_g_0 - 1) # Initial number of Moles of Lithium in the Anode [mol]
C_Li_0 = N_Li_0/Vol_a# initial value for Lithium in the Anode [mol_Li/m^3]

'''
Set up the Half Cells
'''
C6 = Species("C",0,0,C_std,0)    
LiC6 = Species("LiC6",-230.0,-11.2,C_std,0)
Li_plus = Species("Li+",-293.3,49.7,C_std,1)

C6_rxn = Participant(C6,1,C_g)
LiC6_rxn = Participant(LiC6,1,C_Li_0)
Li_plus_rxn = Participant(Li_plus,1,C_Li_plus)

React = [LiC6_rxn]
Prod = [Li_plus_rxn,C6_rxn]

HC_a = Half_Cell(React,Prod,n,T) # Create the Half Cell object
indx_Li_a = 1 # index for solid lithium in the anode in the half cell object

HC_c = Half_Cell(Prod,React,n,T) # Create the Half Cell object
indx_Li_c = 0 # index for solid lithium in the cathode in the half cell object

'''
Set up Electrodes
'''
# For now all that is different is the half cell reation directions
Anode = Electrode(BnF_RT_a,BnF_RT_c,Cap_dl,i_o,A_sg,nuA_nF,HC_a,indx_Li_a)
Cathode = Electrode(BnF_RT_a,BnF_RT_c,Cap_dl,i_o,A_sg,nuA_nF,HC_c,indx_Li_c)

'''
Integration
'''
t_start = 0 # [s]
t_end = 1e-5 # length of time passed in the integration [s]
t_span = [t_start,t_end]
SV_0 = [Phi_dl_0,C_Li_0] #,-Phi_dl_0,C_Li_0] # initial values
N = 1000 # number of time steps

SV = solve_ivp(residual,t_span,SV_0,method='BDF',args=(i_ext,Anode,Cathode,sigma_s,t_s),
               rtol = 1e-5,atol = 1e-8)

'''
Post Processing
'''
## Anode
U_cell_a = np.zeros_like(SV.t)
i_far_a = np.zeros_like(SV.t)
for ind, ele in enumerate(SV.t):
    HC_a.C[1] = SV.y[1,ind]
    U_cell_a[ind] = Half_Cell_Eqlib_Potential(HC_a) # Open Cell Potential [V]
    
    i_far_a[ind] = faradaic_current(i_o,SV.y[0,ind],U_cell_a[ind],BnF_RT_a,BnF_RT_c) # Faradaic Current [A/m^2]

i_dl_a = i_ext/A_sg - i_far_a # Double Layer current [A/m^2]
i_ex = i_ext/A_sg*np.ones_like(SV.t) # External Layer current [A/m^2]

# Double Layer Potential Difference 
plt1 = plt.figure(1)
plt.plot(SV.t,SV.y[0])
#plt.ylim((0,2))
plt.title(r"$\Delta\Phi_{dl}$ for the Anode")
plt.xlabel("time [s]")
plt.ylabel("Change in Potential Across the Double Layer [V]")
plt1.show()

# Current
plt2 = plt.figure(2)
plt.plot(SV.t,i_dl_a,'.-')
plt.plot(SV.t,i_ex)
plt.plot(SV.t,i_far_a)
plt.ylim((-2*abs(i_ext)/A_sg ,1.5*abs(i_ext)/A_sg))
plt.title("Current in the Anode")
plt.xlabel("time [s]")
plt.ylabel("Current [A]")
plt.legend(['DL','EXT','FAR'], loc = 'lower right')
plt2.show()

# The trends for these two plots are correct, but the scale is wrong 
# Concentration
plt3 = plt.figure(3)
plt.plot(SV.t,SV.y[1])
plt.xlabel("time [s]")
plt.ylabel("Concentration of Li in the Anode [mol/m^3]")
plt3.show()

# Open Cell Potential
plt4 = plt.figure(4)
plt.plot(SV.t,U_cell_a)
plt.title("Open Cell Potential for the Anode")
plt.xlabel("time [s]")
plt.ylabel("U [V]")
plt.show()