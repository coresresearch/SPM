# spm.py
#
# This file serves as the main model file.  
# It is called by the user to run the mode
# 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential, residual, Species, Participant, Half_Cell

# LiC6 -> Li+ + C6 + e- (reaction at the anode and cathode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
# The rate Lithium ions enter the electrolyte from the anode is equal to the rate of Li+ enter the seperator and eventually the cathode
# I assume the concentration in the electrolyte is in uniform so there is no diffusion and there is no bulk movement of the fluid
#   so there is no convection therefore there is only migration which is driven by the potential difference
# Potential drop across the electroltye obeys Ohm's Law
# The cathode and anode have the same composition and dimensions 
# I do not create an instance of the species/particiant classes for the electron so I track the 
#   sign of the electron speperately using 'n'

# Since both electrodes are the same, the cell potential (V_cell) is always twice the origonal open cell potential.

'''
USER INPUTS
'''
# 
# Operating Conditions
i_ext = -2000 # external current [A/m^2]
T = 298.15 # standard temperature [K]

# Initial Conditions
Phi_dl_0 = 1.5 #-U+ 0.75 # initial value for Phi_dl [V]
X_Li_0 = 0.4 # Initial Mole Fraction of the Anode for Lithium [-]

# Material parameters:
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
C_Li_plus = 1000 # Concentration of Li+ in the Electrolyte [mol/m^2] (I assume electrolyte transport is fast)
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)
sigma_s = 1.2 # Ionic conductivity for the seperator [1/m-ohm] (this is concentration dependent but it is constant for now)

# Kinetic parameters
i_o = 120 # Exchange Current Density [A/m^2] 
Cap_dl = 6*10**-5 # Double Layer Capacitance [F/m^2]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
# both Li+ and electrons are products in this reaction so they have postive coefficients
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
n = 1 # number of electrons [mol_electrons/mol_rxn]


# Microstructure
t_s =  1e-5 # thickness of the seperator [m]
Delta_y = 50*10**-6 # Anode thickness [m]
r = 5*10**-6 # particle radius [m]
Epsilon_g = 0.65 # volume fraction fo graphite [-]

# Simulation parameters
dPhi_min = 0.2 # Minimum voltage at which to terminate the integration [V]

'''
Constants and Parameters
''' 
F = 96485.34 #Faraday's number [C/mol_electron]
R = 8.3145 #Universal gas constant [J/mol-K]

# Geometry (same for both electrodes)
Vol_a = (4*np.pi/3)*r**3 # Volume of a single anode particle [m^3]
A_surf = 4*np.pi*r**2 # geometric surface area of a single anode particle [m^2]
A_s = 3/r # ratio of surface area to volume for the graphite anode [1/m]
A_sg = Epsilon_g*Delta_y*A_s # interface surface area per geometric surface area [m^2_interface/m^2_geometric]

# Initial Values
X_g_0 = 1 - X_Li_0 # Initial Mole Fraction of Graphite [-]
# The number of moles of C6 in a single anode particle [mol]. Divide by 6 to 
#    convert MW from per mole of carbon to per mole of C6 [mol]
N_g = Vol_a*rho_g/MW_g/6
# Molar concentration of C6 in the anode [mol/m^3]
C_g = N_g/Vol_a
# Initial Lithium molar concentration in the Anode [mol_Li/m^3-graphite]
C_Li_0 = C_g*X_Li_0 

'''
Set up the Half Cells
'''
# Reacation: LiC6 -> C6 + Li+ + e-
C6 = Species("C",0,0,C_g,0)
LiC6 = Species("LiC6",-230000,-11.2,C_g,0)
Li_plus = Species("Li+",-293300,49.7,C_std,1)

C6_rxn = Participant(C6,1,C_g)
LiC6_rxn = Participant(LiC6,1,C_Li_0)
Li_plus_rxn = Participant(Li_plus,1,C_Li_plus)

React = [LiC6_rxn]
Prod = [Li_plus_rxn,C6_rxn]

# Both electrodes have the same reation and in this forward reaction one electron is 
#   produced so n = 1 (as it is defined above in the inputs)
Anode = Half_Cell(React,Prod,n,T,Beta,F,R,Cap_dl,i_o,A_sg,A_s,'LiC6','Li+')
Cathode = Half_Cell(React,Prod,n,T,Beta,F,R,Cap_dl,i_o,A_sg,A_s,'LiC6','Li+')

'''
Integration
'''
t_start = 0 # [s]
t_end = 3e1 # length of time passed in the integration [s]
t_span = [t_start,t_end]
SV_0 = [Phi_dl_0,C_Li_0,-Phi_dl_0,C_Li_0] # initial values


def min_voltage(_,SV,i_ext,Anode,Cathode):
    return SV[0] - dPhi_min

min_voltage.terminal = True

SV = solve_ivp(residual,t_span,SV_0,method='BDF',
               args=(i_ext,Anode,Cathode),
               rtol = 1e-5,atol = 1e-8, events=min_voltage)

'''
Post Processing
'''
## Anode LiC6 -> Li+ + C6 + e- 
U_cell_a = np.zeros_like(SV.t)
i_far_a = np.zeros_like(SV.t)
for ind, ele in enumerate(SV.t):
    # I find open cell potential and faradic current using the same process as the residual function
    Anode.C[0] = SV.y[1,ind]/Anode.C_int[0]
    Anode.C[-1] = 1. - Anode.C[0]
    U_cell_a[ind] = Half_Cell_Eqlib_Potential(Anode) # Open Cell Potential [V]
    
    i_far_a[ind] = faradaic_current(i_o,SV.y[0,ind],U_cell_a[ind],Anode.BnF_RT_a,Anode.BnF_RT_c) # Faradaic Current [A/m^2]

i_dl_a = i_ext/A_sg - i_far_a # Double Layer current [A/m^2]

## Cathode LiC6 -> Li+ + C6 + e- 
U_cell_c = np.zeros_like(SV.t)
i_far_c = np.zeros_like(SV.t)
for ind, ele in enumerate(SV.t):
    Cathode.C[0] = SV.y[3,ind]/Cathode.C_int[0]
    Cathode.C[-1] = 1. - Cathode.C[0]
    U_cell_c[ind] = Half_Cell_Eqlib_Potential(Cathode) # Open Cell Potential [V]   
    
    i_far_c[ind] = faradaic_current(i_o,SV.y[2,ind],U_cell_c[ind],Cathode.BnF_RT_a,Cathode.BnF_RT_c) # Faradaic Current [A/m^2]
i_dl_c = i_ext/A_sg + i_far_c # Double Layer current [A/m^2]

## Seperator
i_ex = i_ext/A_sg*np.ones_like(SV.t) # External current as an array [A/m^2]
Delta_Phi_s = -i_ex*t_s/sigma_s # Potential drop across the seperator [V]

## Whole Cell
# Phi_c - Phi_el_c = delta_Phi_dl_c  
# Phi_a - Phi_el_a = delta_Phi_dl_a ; 

# V_cell = Phi_c - Phi_a 
# V_cell = Ph_c - Phi_elc_c - Phi_elc_a - Phi_a

# V_cell = delta_Phi_dl_c + delta_Phi_dl_a
V_cell = SV.y[2] - SV.y[0]

## Tracking moles of lithium
mol_a = Vol_a*SV.y[1] # moles of Lithium in the Anode
mol_c = Vol_a*SV.y[3] # moles of Lithium in the Cathode
mol_t = mol_c + mol_a # total moles of Lithium

'''
Plotting
'''
# Double Layer Potential Difference 
plt1 = plt.figure(1)
plt.plot(SV.t,SV.y[0])
plt.plot(SV.t,SV.y[2])
plt.title(r"$\Delta\Phi_{dl}$")
plt.xlabel("time [s]")
plt.ylabel("Change in Potential Across the Double Layer [V]")
plt.legend(['Anode','Cathode'], loc = 'upper right')

# Current to the Anode
plt2 = plt.figure(2)
plt.plot(SV.t,i_dl_a,'.-')
plt.plot(SV.t, i_ex, linewidth=4)
plt.plot(SV.t,i_far_a, color='black')
plt.ylim((-2*abs(i_ext)/A_sg ,1.5*abs(i_ext)/A_sg))
plt.title("Current in the Anode")
plt.xlabel("time [s]")
plt.ylabel("Current [A]")
plt.legend(['DL','EXT','FAR'], loc = 'lower right')

# Current to the Cathode
plt3 = plt.figure(3)
plt.plot(SV.t,i_dl_c,'.-')
plt.plot(SV.t, i_ex, linewidth=4)
plt.plot(SV.t,i_far_c, color='black')
plt.ylim((-2*abs(i_ext)/A_sg ,1.5*abs(i_ext)/A_sg))
plt.title("Current in the Cathode")
plt.xlabel("time [s]")
plt.ylabel("Current [A]")
plt.legend(['DL','EXT','FAR'], loc = 'lower right')

# Amount of Lithium
plt7 = plt.figure(7)
plt.plot(SV.t,mol_a)
plt.plot(SV.t,mol_c)
plt.plot(SV.t,mol_t,'--')
plt.legend(['Anode','Cathode','Total'], loc = 'lower left')
plt.title("Amount of Lithium")
plt.xlabel("time [s]")
plt.ylabel("Lithium [mol]")

# Open Cell Potential
plt5 = plt.figure(5)
plt.plot(SV.t,U_cell_a)
plt.plot(SV.t,U_cell_c)
plt.title("Open Cell Potential")
plt.xlabel("time [s]")
plt.ylabel("Equilibrium Potential [V]")
plt.legend(['Anode','Cathode'], loc = 'upper left')

# Cell Voltage
plt6 = plt.figure(6)
plt.plot(SV.t,SV.y[0],'--')
plt.plot(SV.t,SV.y[2],'--')
plt.plot(SV.t,Delta_Phi_s,'--')
plt.plot(SV.t,V_cell)
plt.legend([r'$\Delta\Phi_{dl_a}$',r'$\Delta\Phi_{dl_c}$',r'$\Delta\Phi_{sep}$', r'$V_{cell}$'], loc = 'best')
plt.title("Vell Voltage")
plt.xlabel("time [s]")
plt.ylabel("Voltage [V]")

plt.show()

