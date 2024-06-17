# spm.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential, residual, Species, Participant, Half_Cell

# Anode is on the left at x=0 and Cathode is on the right
# LiC6 -> Li+ + C6 + e- (reaction at the anode and cathode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
# The rate Lithium ions enter the electrolyte from the anode is equal to the rate of Li+ enter the seperator and eventually the cathode
# I assume the concentration in the electrolyte is in uniform so there is no diffusion and there is no bulk movement of the fluid
#   so there is no convection therefore there is only migration which is driven by the potential difference
# Potential drop across the electroltye obeys Ohm's Law
# The cathode and anode have the same composition and dimensions 
# I do not create an instance of the species/particiant classes for the electron so I track the 
#   sign of the electron speperately using 'n'

'''
USER INPUTS
'''
# Operating Conditions
i_ext = -2000 # external current into the Anode [A/m^2] 
T = 298.15 # standard temperature [K]

# Initial Conditions
Phi_dl_0_an = .6 # initial value for Phi_dl for the Anode [V]
X_Li_0_an = 0.5 # Initial Mole Fraction of the Anode for Lithium [-]
Phi_dl_0_ca = .3 # initial value for Phi_dl for the Cathode [V]
X_Li_0_ca = 0.5 # Initial Mole Fraction of the Cathode for Lithium [-]

# Material parameters:
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
C_Li_plus = 1000 # Concentration of Li+ in the Electrolyte [mol/m^2] (I assume electrolyte transport is fast)
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)
sigma_sep = 1.2 # Ionic conductivity for the seperator [1/m-ohm] (this is concentration dependent but it is constant for now)

# Kinetic parameters (both electrodes have the same reaction for now)
i_o = 120 # Exchange Current Density [A/m^2] 
Cap_dl = 6*10**-5 # Double Layer Capacitance [F/m^2]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
# both Li+ and electrons are products in this reaction so they have postive coefficients
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
n = 1 # number of electrons [mol_electrons/mol_rxn]

# Microstructure
t_sep =  1e-5 # thickness of the seperator [m]
Delta_y_an = 50*10**-6 # Anode thickness [m]
r_an = 5*10**-6 # Anode particle radius [m]
Delta_y_ca = 50*10**-6 # Cathode thickness [m]
r_ca = 5*10**-6 # Cathode particle radius [m]
Epsilon_g = 0.65 # volume fraction fo graphite [-]

# Simulation parameters
V_min = -0.7 # Minimum cell voltage at which to terminate the integration [V]
V_max = 0.7 # Maximum cell voltage at which to terminate the integration [V]

'''
Constants and Parameters
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

# Geometry
#   Anode
Vol_an = (4*np.pi/3)*r_an**3 # Volume of a single anode particle [m^3]
A_surf_an = 4*np.pi*r_an**2 # geometric surface area of a single anode particle [m^2]
A_s_an = 3/r_an # ratio of surface area to volume for the graphite anode [1/m]
A_sg_an = Epsilon_g*Delta_y_an*A_s_an # interface surface area per geometric surface area [m^2_interface/m^2_geometric]
#   Cathode
Vol_ca = (4*np.pi/3)*r_ca**3 # Volume of a single anode particle [m^3]
A_surf_ca = 4*np.pi*r_ca**2 # geometric surface area of a single anode particle [m^2]
A_s_ca = 3/r_ca # ratio of surface area to volume for the graphite anode [1/m]
A_sg_ca = Epsilon_g*Delta_y_ca*A_s_ca # interface surface area per geometric surface area [m^2_interface/m^2_geometric]

# Initial Concentrations
# The number of moles of C6 in a single anode particle [mol]. Divide by 6 to 
#    convert MW from per mole of carbon to per mole of C6 [mol]
C_g = rho_g/MW_g/6 # Molar concentration of C6 in the anode [mol/m^3]

X_g_0_an = 1 - X_Li_0_an # Initial Mole Fraction of Graphite [-]
C_Li_0_an = C_g*X_Li_0_an # Initial Lithium molar concentration in the Anode [mol_Li/m^3-graphite]
C_g_0_an = C_g*X_g_0_an # Initial graphite molar concentration in the Anode [mol_Li/m^3-graphite]

X_g_0_ca = 1 - X_Li_0_ca # Initial Mole Fraction of Graphite [-]
C_Li_0_ca = C_g*X_Li_0_ca # Initial Lithium molar concentration in the Anode [mol_Li/m^3-graphite]
C_g_0_ca = C_g*X_g_0_ca # Initial graphite molar concentration in the Anode [mol_Li/m^3-graphite]

'''
Set up the Half Cells
'''
# The initial concentrations of LiC6 and C6 depend on the user defined initial mole fractions, 
#   the concentraions of Li+ in the electroltyes is static and the same in both half cells

# Reacation: LiC6 -> C6 + Li+ + e-
C6 = Species("C",0,0,C_g,0)
LiC6 = Species("LiC6",-230000,-11.2,C_g,0)
Li_plus = Species("Li+",-293300,49.7,C_std,1)

C6_rxn_an = Participant(C6,1,C_g_0_an)
LiC6_rxn_an = Participant(LiC6,1,C_Li_0_an)
Li_plus_rxn_an = Participant(Li_plus,1,C_Li_plus)

React_an = [LiC6_rxn_an]
Prod_an = [Li_plus_rxn_an,C6_rxn_an]

# Both electrodes have the same reation and in this forward reaction one electron is 
#   produced so n = 1 (as it is defined above in the inputs)
Anode = Half_Cell(React_an,Prod_an,n,T,Beta,F,R,Cap_dl,i_o,A_sg_an,A_s_an,'LiC6','Li+')

# Reacation: LiC6 -> C6 + Li+ + e-
C6_rxn_ca = Participant(C6,1,C_g_0_ca)
LiC6_rxn_ca = Participant(LiC6,1,C_Li_0_ca)
Li_plus_rxn_ca = Participant(Li_plus,1,C_Li_plus)

React_ca = [LiC6_rxn_ca]
Prod_ca = [Li_plus_rxn_ca,C6_rxn_ca]

Cathode = Half_Cell(React_ca,Prod_ca,n,T,Beta,F,R,Cap_dl,i_o,A_sg_ca,A_s_ca,'LiC6','Li+')

'''
Integration
'''
t_start = 0 # [s]
t_end = 3e1 # length of time passed in the integration [s]
t_span = [t_start,t_end]
SV_0 = [Phi_dl_0_an,C_Li_0_an,Phi_dl_0_ca,C_Li_0_ca] # initial values

# These are place holders for now, I will change to V_max and V_min soon
def min_voltage(_,SV,i_ext,Anode,Cathode):
    V_cell = SV[2] + i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_min

min_voltage.terminal = True

def max_voltage(_,SV,i_ext,Anode,Cathode):
    V_cell = SV[2] + i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_max

max_voltage.terminal = True

SV = solve_ivp(residual,t_span,SV_0,method='BDF',
               args=(i_ext,Anode,Cathode),
               rtol = 1e-5,atol = 1e-8, events=(min_voltage,max_voltage))

'''
Post Processing
'''
## Anode LiC6 -> Li+ + C6 + e- 
U_cell_an = np.zeros_like(SV.t)
i_far_an = np.zeros_like(SV.t)
for ind, ele in enumerate(SV.t):
    # I find open cell potential and faradic current using the same process as the residual function
    Anode.C[0] = SV.y[1,ind]/Anode.C_int[0]
    Anode.C[-1] = 1. - Anode.C[0]
    U_cell_an[ind] = Half_Cell_Eqlib_Potential(Anode) # Open Cell Potential [V]
    
    i_far_an[ind] = faradaic_current(i_o,SV.y[0,ind],U_cell_an[ind],Anode.BnF_RT_an,Anode.BnF_RT_ca) # Faradaic Current [A/m^2]

i_dl_an = i_ext/A_sg_an - i_far_an # Double Layer current [A/m^2]

## Cathode LiC6 -> Li+ + C6 + e- 
U_cell_ca = np.zeros_like(SV.t)
i_far_ca = np.zeros_like(SV.t)
for ind, ele in enumerate(SV.t):
    Cathode.C[0] = SV.y[3,ind]/Cathode.C_int[0]
    Cathode.C[-1] = 1. - Cathode.C[0]
    U_cell_ca[ind] = Half_Cell_Eqlib_Potential(Cathode) # Open Cell Potential [V]   
    
    i_far_ca[ind] = faradaic_current(i_o,SV.y[2,ind],U_cell_ca[ind],Cathode.BnF_RT_an,Cathode.BnF_RT_ca) # Faradaic Current [A/m^2]
    
i_dl_ca = i_ext/A_sg_ca + i_far_ca # Double Layer current [A/m^2]

## Seperator (Phi_el_ca - Ph_el_an)
i_ex = i_ext*np.ones_like(SV.t) # External current as an array [A/m^2]
Delta_Phi_sep = i_ex*t_sep/sigma_sep # Potential drop across the seperator [V]

## Whole Cell
# Phi_ca - Phi_el_ca = Delta_Phi_dl_ca  
# Phi_an - Phi_el_an = Delta_Phi_dl_an 

# V_cell = Phi_ca - Phi_an 
# V_cell = Ph_ca - Phi_elc_ca + Phi_elc_ca - Phi_elc_an + Phi_elc_an - Phi_an
# V_cell = (Ph_ca - Phi_elc_ca) + (Phi_elc_ca - Phi_elc_an) - (Phi_an - Phi_el_an)

# V_cell = Delta_Phi_dl_ca + Delta_Phi_sep - Delta_Phi_dl_an
V_cell = SV.y[2] + Delta_Phi_sep - SV.y[0]

## Tracking moles of lithium
mol_an = Vol_an*SV.y[1] # moles of Lithium in the Anode
mol_ca = Vol_an*SV.y[3] # moles of Lithium in the Cathode
mol_total = mol_ca + mol_an # total moles of Lithium

'''
Plotting
'''
fig1, (ax1, ax2) = plt.subplots(2)
# Current to the Anode
ax1.plot(SV.t,i_dl_an,'.-', color='firebrick')
ax1.plot(SV.t, i_ex/A_sg_an, linewidth=4, color='deepskyblue')
ax1.plot(SV.t,i_far_an, color='black')
ax1.set_ylim((-2*abs(i_ext)/A_sg_an ,1.5*abs(i_ext)/A_sg_an))
ax1.set_title("Current in the Anode")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("Current [A/m^2]")
ax1.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5), loc = 'center left')

# Current to the Cathode
ax2.plot(SV.t,i_dl_ca,'.-', color='firebrick')
ax2.plot(SV.t, i_ex/A_sg_ca, linewidth=4, color='deepskyblue')
ax2.plot(SV.t,i_far_ca, color='black')
ax2.set_ylim((-2*abs(i_ext)/A_sg_ca ,1.5*abs(i_ext)/A_sg_ca))
ax2.set_title("Current in the Cathode")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Current [A/m^2]")
ax2.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5),loc = 'center left')
fig1.tight_layout()

fig2, (ax3, ax4) = plt.subplots(2)
# Double Layer Potential Difference 
ax3.plot(SV.t,SV.y[0])
ax3.plot(SV.t,SV.y[2])
ax3.set_title(r"$\Delta\Phi_{dl}$")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Change in Potential [V]")
ax3.legend(['Anode','Cathode'], loc = 'best')

# Open Cell Potential
ax4.plot(SV.t,U_cell_an)
ax4.plot(SV.t,U_cell_ca)
ax4.set_title("Open Cell Potential")
ax4.set_xlabel("time [s]")
ax4.set_ylabel("Equilibrium Potential [V]")
ax4.legend(['Anode','Cathode'], loc = 'best')
fig2.tight_layout()

# Amount of Lithium
plt3 = plt.figure(3)
plt.plot(SV.t,mol_an)
plt.plot(SV.t,mol_ca)
plt.plot(SV.t,mol_total,'--')
plt.legend(['Anode','Cathode','Total'], loc = 'best')
plt.title("Amount of Lithium")
plt.xlabel("time [s]")
plt.ylabel("Lithium [mol]")

# Cell Voltage
plt4 = plt.figure(4)
plt.plot(SV.t,SV.y[0],'--')
plt.plot(SV.t,SV.y[2],'--')
plt.plot(SV.t,Delta_Phi_sep,'--')
plt.plot(SV.t,V_cell)
plt.legend([r'$\Delta\Phi_{dl,a}$',r'$\Delta\Phi_{dl,c}$',r'$\Delta\Phi_{sep}$', r'$V_{cell}$'], loc = 'best')
plt.title("Cell Voltage")
plt.xlabel("time [s]")
plt.ylabel("Voltage [V]")

plt.show()