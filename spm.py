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
# Galvanostatic

'''
USER INPUTS
'''
# Simulation parameters
# The volatage limits are options, but I am setting them so high they don't trip.
#   Mole fractions are the functional limiting events right now 
V_min = -10 # Minimum cell voltage at which to terminate the integration [V]
V_max = 10 # Maximum cell voltage at which to terminate the integration [V]
X_Li_min = 0.01 # Lithium mole fraction in an electrode at which to terminate the integration [-]
X_Li_max = 0.99 # Lithium mole fraction in an electrode at which to terminate the integration [-]

# Operating Conditions
# If there is more than one external current the simulation will run back to back
i_external = np.array([0, 2000, -3000, 1000,-500]) # external current into the Anode [A/m^2] 
t_sim_max = [5,15,5,30,10] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

# Initial Conditions
Phi_dl_0_an = 0.5 # initial value for Phi_dl for the Anode [V]
X_Li_0_an = 0.35 # Initial Mole Fraction of the Anode for Lithium [-]
Phi_dl_0_ca = -0.6 # initial value for Phi_dl for the Cathode [V]
X_Li_0_ca = 0.72 # Initial Mole Fraction of the Cathode for Lithium [-]

# Material parameters:
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
MW_FePO4 = 150.815 # Molectular Weight of iron phosphate [g/mol]
rho_FePO4 = 2.87e6 # Denstity of iron phosphate [g/m^3]
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
Delta_y_an = 25*10**-6 # Anode thickness [m]
r_an = 5*10**-6 # Anode particle radius [m]
Delta_y_ca = 50*10**-6 # Cathode thickness [m]
r_ca = 5*10**-6 # Cathode particle radius [m]
Epsilon_g = 0.65 # volume fraction fo graphite [-]

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

C_FePO4 = rho_FePO4/MW_FePO4 # Molar concentration of FePO4 in the cathode [mol/m^3]

X_FePO4_0 = 1 - X_Li_0_ca # Initial Mole Fraction of FePO4 [-]
C_Li_0_ca = C_FePO4*X_Li_0_ca # Initial Lithium molar concentration in the Anode [mol_Li/m^3-FePO4]
C_FePO4_0 = C_FePO4*X_FePO4_0 # Initial FePO4 molar concentration in the Anode [mol_Li/m^3-FePO4]

'''
Set up the Half Cells
'''
# The initial concentrations of LiC6 and C6 depend on the user defined initial mole fractions, 
#   the concentraions of Li+ in the electroltyes are static and the same in both half cells
# I am not confident in my standard entropy values for the cathode, but this is not an issue
#   yet because the cell is still at standard temperature

# Anode reacation: LiC6 -> C6 + Li+ + e-
LiC6 = Species("LiC6",-230000,-11.2,C_g,0)
C6 = Species("C",0,0,C_g,0)
Li_plus = Species("Li+",-293300,49.7,C_std,1)

LiC6_rxn_an = Participant(LiC6,1,C_Li_0_an)
C6_rxn_an = Participant(C6,1,C_g_0_an)
Li_plus_rxn_an = Participant(Li_plus,1,C_Li_plus)

React_an = [LiC6_rxn_an]
Prod_an = [Li_plus_rxn_an,C6_rxn_an]

# Both electrodes have the same reation and in this forward reaction one electron is 
#   produced so n = 1 (as it is defined above in the inputs)
Anode = Half_Cell(React_an,Prod_an,n,T,Beta,F,R,Cap_dl,i_o,A_sg_an,A_s_an,'LiC6','Li+')

# Cathode reacation: LiFePO4 -> FePO4 + Li+ + e-
LiFePO4 = Species("LiFePO4",-326650,130.95,C_FePO4,0)
FePO4 = Species("FePO4",0,171.3,C_FePO4,0)

LiFePO4_rxn_ca = Participant(LiFePO4,1,C_Li_0_ca)
FePO4_rxn_ca = Participant(FePO4,1,C_FePO4_0)
Li_plus_rxn_ca = Participant(Li_plus,1,C_Li_plus)

React_ca = [LiFePO4_rxn_ca]
Prod_ca = [Li_plus_rxn_ca,FePO4_rxn_ca]

Cathode = Half_Cell(React_ca,Prod_ca,n,T,Beta,F,R,Cap_dl,i_o,A_sg_ca,A_s_ca,'LiFePO4','Li+')

'''
Integration
'''
# Integration Limits
def min_voltage(_,SV,i_ext,Anode,Cathode):
    V_cell = SV[2] + i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_min
min_voltage.terminal = True

def max_voltage(_,SV,i_ext,Anode,Cathode):
    V_cell = SV[2] + i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_max
max_voltage.terminal = True

def min_Li_an(_,SV,i_ext,Anode,Cathode):
    return SV[1]/Anode.C_int[Anode.ind_track] - X_Li_min
min_Li_an.terminal = True

def max_Li_an(_,SV,i_ext,Anode,Cathode):
    return SV[1]/Anode.C_int[Anode.ind_track] - X_Li_max
max_Li_an.terminal = True

def min_Li_ca(_,SV,i_ext,Anode,Cathode):
    return SV[3]/Cathode.C_int[Cathode.ind_track] - X_Li_min
min_Li_ca.terminal = True

def max_Li_ca(_,SV,i_ext,Anode,Cathode):
    return SV[3]/Cathode.C_int[Cathode.ind_track] - X_Li_max
max_Li_ca.terminal = True
    
sim_inputs = [Phi_dl_0_an, C_Li_0_an, Phi_dl_0_ca, C_Li_0_ca, 0] # [SV[0], SV[1], SV[2], SV[3], time]

for ind, i_ext in enumerate(i_external):
 
    # Integration parameters 
    t_start = sim_inputs[-1] # [s]
    t_end = t_start + t_sim_max[ind] # max length of time passed in the integration [s]
    t_span = [t_start,t_end]
    SV_0 = [sim_inputs[0], sim_inputs[1], sim_inputs[2], sim_inputs[3]] # initial values

    # Integrater
    SV = solve_ivp(residual,t_span,SV_0,method='BDF',
                args=(i_ext,Anode,Cathode),
                rtol = 1e-8,atol = 1e-10, events=(min_voltage, max_voltage, min_Li_an, max_Li_an, min_Li_ca, max_Li_ca))
    
    new_outputs = np.array([SV.y[0], SV.y[1], SV.y[2], SV.y[3], SV.t, i_ext*np.ones_like(SV.t)])
    if ind == 0:
        sim_outputs = new_outputs
    else:
        sim_outputs = np.concatenate((sim_outputs,new_outputs), axis = 1)

    # Once an event is triggered the intergration stops, I want it to just swtich to the next current
    #   so I move the inputs just under the event threshold (back one index) so the integration can proceed 
    if SV.status == 1:
        sim_inputs = [SV.y[0,-2], SV.y[1,-2], SV.y[2,-2], SV.y[3,-2], SV.t[-2]]
    else:
        sim_inputs = [SV.y[0,-1], SV.y[1,-1], SV.y[2,-1], SV.y[3,-1], SV.t[-1]]
    
'''
Post Processing
'''
[Delta_Phi_dl_an, C_Li_an, Delta_Phi_dl_ca, C_Li_ca, time, i_ex] = sim_outputs

## Anode LiC6 -> Li+ + C6 + e- 
U_cell_an = np.zeros_like(sim_outputs[0])
i_far_an = np.zeros_like(sim_outputs[0])
for ind, ele in enumerate(Delta_Phi_dl_an):
    # I find open cell potential and faradic current using the same process as the residual function
    Anode.C[0] = C_Li_an[ind]/Anode.C_int[0]
    Anode.C[-1] = 1. - Anode.C[0]
    U_cell_an[ind] = Half_Cell_Eqlib_Potential(Anode) # Open Cell Potential [V]
    
    i_far_an[ind] = faradaic_current(i_o,sim_outputs[0,ind],U_cell_an[ind],Anode.BnF_RT_an,Anode.BnF_RT_ca) # Faradaic Current [A/m^2]

i_dl_an = i_ex/A_sg_an - i_far_an # Double Layer current [A/m^2]

## Cathode LiFePO4 -> FePO4 + Li+ + e-
U_cell_ca = np.zeros_like(sim_outputs[0])
i_far_ca = np.zeros_like(sim_outputs[0])
for ind, ele in enumerate(Delta_Phi_dl_ca):
    Cathode.C[0] = C_Li_ca[ind]/Cathode.C_int[0]
    Cathode.C[-1] = 1. - Cathode.C[0]
    U_cell_ca[ind] = Half_Cell_Eqlib_Potential(Cathode) # Open Cell Potential [V]   
    
    i_far_ca[ind] = faradaic_current(i_o,sim_outputs[2,ind],U_cell_ca[ind],Cathode.BnF_RT_an,Cathode.BnF_RT_ca) # Faradaic Current [A/m^2]
    
i_dl_ca = -i_ex/A_sg_ca - i_far_ca # Double Layer current [A/m^2]

## Seperator (Delta_Phi_sep = Phi_el_ca - Ph_el_an)
# A negative external current to the anode should lead ot Li+ ions traveling from the anode 
#   to the cathode. In this case, there should be a potential drop due to the ohmic resistance
#   which would make Phi_el_ca < Ph_el_an and therefore Delta_Phi_sep < 0.
# Long way of saying Delta_Phi_sep should have the same sign as i_ext
Delta_Phi_sep = i_ex*t_sep/sigma_sep # Potential drop across the seperator [V]

## Whole Cell
# Phi_ca - Phi_el_ca = Delta_Phi_dl_ca  
# Phi_an - Phi_el_an = Delta_Phi_dl_an 

# V_cell = Phi_ca - Phi_an 
# V_cell = Ph_ca - Phi_elc_ca + Phi_elc_ca - Phi_elc_an + Phi_elc_an - Phi_an
# V_cell = (Ph_ca - Phi_elc_ca) + (Phi_elc_ca - Phi_elc_an) - (Phi_an - Phi_el_an)

V_cell = Delta_Phi_dl_ca + Delta_Phi_sep - Delta_Phi_dl_an

## Tracking moles of lithium
mol_an = Delta_y_an*C_Li_an # moles of Lithium in the Anode per unit area [moles/m^2]
mol_ca = Delta_y_ca*C_Li_ca # moles of Lithium in the Cathode per unit area [moles/m^2]
mol_total = mol_ca + mol_an # total moles of Lithium

'''
Plotting
'''
fig1, (ax1, ax2) = plt.subplots(2)
# Current to the Anode
ax1.plot(time,i_dl_an,'.-', color='firebrick')
ax1.plot(time, i_ex/A_sg_an, linewidth=4, color='deepskyblue')
ax1.plot(time,i_far_an, color='black')
ax1.set_ylim((-2*max(abs(i_external))/A_sg_an ,1.5*max(abs(i_external))/A_sg_an))
ax1.set_title("Current in the Anode")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("Current [A/m^2]")
ax1.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5), loc = 'center left')

# Current to the Cathode
ax2.plot(time,i_dl_ca,'.-', color='firebrick')
ax2.plot(time, i_ex/A_sg_ca, linewidth=4, color='deepskyblue')
ax2.plot(time,i_far_ca, color='black')
ax2.set_ylim((-2*max(abs(i_external))/A_sg_an ,1.5*max(abs(i_external))/A_sg_an))
ax2.set_title("Current in the Cathode")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("Current [A/m^2]")
ax2.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5),loc = 'center left')
fig1.tight_layout()

fig2, (ax3, ax4) = plt.subplots(2)
# Double Layer Potential Difference 
ax3.plot(time,Delta_Phi_dl_an)
ax3.plot(time,sim_outputs[2])
ax3.set_title(r"$\Delta\Phi_{dl}$")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Change in Potential [V]")
ax3.legend(['Anode','Cathode'], loc = 'best')

# Open Cell Potential
ax4.plot(time,U_cell_an)
ax4.plot(time,U_cell_ca)
ax4.set_title("Open Cell Potential")
ax4.set_xlabel("time [s]")
ax4.set_ylabel("Equilibrium Potential [V]")
ax4.legend(['Anode','Cathode'], loc = 'best')
fig2.tight_layout()

fig3, (ax5, ax6) = plt.subplots(2)
# Amount of Lithium
ax5.plot(time,mol_an)
ax5.plot(time,mol_ca)
ax5.plot(time,mol_total,'--')
ax5.legend(['Anode','Cathode','Total'], bbox_to_anchor=(1, 0.5),loc = 'center left')
ax5.set_title("Amount of Lithium")
ax5.set_xlabel("time [s]")
ax5.set_ylabel("Lithium [mol/m^2]")

# Concentration of Lithium
ax6.plot(time,C_Li_an/Anode.C_int[Anode.ind_track])
ax6.plot(time,C_Li_ca/Anode.C_int[Anode.ind_track])
ax6.plot(time,np.ones_like(C_Li_ca),'--','r')
ax6.plot(time,np.zeros_like(C_Li_ca),'--','r')
ax6.legend(['Anode','Cathode'], bbox_to_anchor=(1, 0.5),loc = 'center left')
ax6.set_title("Concentration of Lithium")
ax6.set_xlabel("time [s]")
ax6.set_ylabel("Lithium [mol/m^3]")
fig3.tight_layout()

# Cell Voltage
plt4 = plt.figure(4)
plt.plot(time,Delta_Phi_dl_an,'--')
plt.plot(time,sim_outputs[2],'--')
plt.plot(time,Delta_Phi_sep,'--')
plt.plot(time,V_cell)
plt.legend([r'$\Delta\Phi_{dl,a}$',r'$\Delta\Phi_{dl,c}$',r'$\Delta\Phi_{sep}$', r'$V_{cell}$'], bbox_to_anchor=(1, 0.5),loc = 'center left')
plt.title("Cell Voltage")
plt.xlabel("time [s]")
plt.ylabel("Voltage [V]")
plt.tight_layout()

plt.show()