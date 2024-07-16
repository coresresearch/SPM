# single_electrode.py
# I want to discritize a single electrode on its own before incorperating it into the model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential, residual_single, Species, Participant, Half_Cell

'''
USER INPUTS
'''
# Simulation parameters
# The volatage limits are options, but I am setting them so high they don't trip.
#   Mole fractions are the functional limiting events right now 
X_Li_min = 0.01 # Lithium mole fraction in an electrode at which to terminate the integration [-]
X_Li_max = 0.99 # Lithium mole fraction in an electrode at which to terminate the integration [-]

# Operating Conditions
# If there is more than one external current the simulation will run back to back
i_external = np.array([0]) # external current into the Anode [A/m^2] 
t_sim_max = [1e-7] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

# Initial Conditions
Phi_dl_0_an = -0.64 # initial value for Phi_dl for the Anode [V]
X_Li_0_an = 0.35 # Initial Mole Fraction of the Anode for Lithium [-]

# Material parameters:
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
C_Li_plus = 1000 # Concentration of Li+ in the Electrolyte [mol/m^3] (I assume electrolyte transport is fast)
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)

# Activity Coefficients (All 1 for now)
gamma_LiC6 = 1
gamma_C6 = 1
gamma_Li_plus = 1

# Kinetic parameters
i_o_reff = 120 # Exchange Current Density [A/m^2] 
Cap_dl = 6*10**-5 # Double Layer Capacitance [F/m^2]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
# both Li+ and electrons are products in this reaction so they have postive coefficients
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
n = -1 # number of electrons multiplied by the charge of an electron [mol_electrons/mol_rxn]

# Microstructure
Delta_y_an = 25*10**-6 # Anode thickness [m]
r_an = 5*10**-6 # Anode particle radius [m]
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

# Initial Concentrations
# The number of moles of C6 in a single anode particle [mol]. Divide by 6 to 
#    convert MW from per mole of carbon to per mole of C6 [mol]
C_g = rho_g/MW_g/6 # Molar concentration of C6 in the anode [mol/m^3]

X_g_0_an = 1 - X_Li_0_an # Initial Mole Fraction of Graphite [-]
C_Li_0_an = C_g*X_Li_0_an # Initial Lithium molar concentration in the Anode [mol_Li/m^3-graphite]
C_g_0_an = C_g*X_g_0_an # Initial graphite molar concentration in the Anode [mol_Li/m^3-graphite]

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

LiC6_rxn_an = Participant(LiC6,1,C_Li_0_an,gamma_LiC6)
C6_rxn_an = Participant(C6,1,C_g_0_an,gamma_C6)
Li_plus_rxn_an = Participant(Li_plus,1,C_Li_plus,gamma_Li_plus)

React_an = [LiC6_rxn_an]
Prod_an = [Li_plus_rxn_an,C6_rxn_an]

# Both electrodes have the same reation and in this forward reaction one electron is 
#   produced so n = 1 (as it is defined above in the inputs)
Anode = Half_Cell(React_an,Prod_an,n,T,Beta,F,R,Cap_dl,i_o_reff,A_sg_an,A_s_an,'LiC6','Li+')

'''
Integration
'''
# Integration Limits
def min_Li_an(_,SV,i_ext,Anode):
    return SV[1]/Anode.C_int[Anode.ind_track] - X_Li_min
min_Li_an.terminal = True

def max_Li_an(_,SV,i_ext,Anode):
    return SV[1]/Anode.C_int[Anode.ind_track] - X_Li_max
max_Li_an.terminal = True

    
sim_inputs = [Phi_dl_0_an, C_Li_0_an, 0] # [SV[0], SV[1], time]

for ind, i_ext in enumerate(i_external):
 
    # Integration parameters 
    t_start = sim_inputs[-1] # [s]
    t_end = t_start + t_sim_max[ind] # max length of time passed in the integration [s]
    t_span = [t_start,t_end]
    SV_0 = [sim_inputs[0], sim_inputs[1]] # initial values

    # Integrater
    SV = solve_ivp(residual_single,t_span,SV_0,method='BDF',
                args=(i_ext,Anode),
                rtol = 1e-8,atol = 1e-10, events=(min_Li_an, max_Li_an))
    
    new_outputs = np.array([SV.y[0], SV.y[1], SV.t, i_ext*np.ones_like(SV.t)])
    if ind == 0:
        sim_outputs = new_outputs
    else:
        sim_outputs = np.concatenate((sim_outputs,new_outputs), axis = 1)

    # Once an event is triggered the intergration stops, I want it to just swtich to the next current
    #   so I move the inputs just under the event threshold (back one index) so the integration can proceed 
    if SV.status == 1:
        sim_inputs = [SV.y[0,-2], SV.y[1,-2], SV.t[-2]]
    else:
        sim_inputs = [SV.y[0,-1], SV.y[1,-1], SV.t[-1]]
    
'''
Post Processing
'''
[Delta_Phi_dl_an, C_Li_an, time, i_ex] = sim_outputs

## Anode LiC6 -> Li+ + C6 + e- 
U_cell_an = np.zeros_like(sim_outputs[0])
i_far_an = np.zeros_like(sim_outputs[0])
i_o_an = np.zeros_like(sim_outputs[0])
for ind, ele in enumerate(Delta_Phi_dl_an):
    # I find open cell potential and faradic current using the same process as the residual function
    X_Li_a = C_Li_an[ind]/Anode.C_int[Anode.ind_track] 
    Anode.activity[Anode.ind_track] = Anode.gamma[Anode.ind_track]*(X_Li_a) 
    Anode.activity[-1] = Anode.gamma[-1]*(1 - X_Li_a) 
    U_cell_an[ind] = Half_Cell_Eqlib_Potential(Anode) # Open Cell Potential [V]
    
    i_o_an[ind] = ((Anode.activity[Anode.ind_track])**Anode.Beta)*(
        (Anode.activity[Anode.ind_ion]*Anode.activity[-1])**(1-Anode.Beta))*Anode.i_o_reff
    
    i_far_an[ind] = faradaic_current(i_o_an[ind],sim_outputs[0,ind],U_cell_an[ind],Anode.BnF_RT_an,Anode.BnF_RT_ca) # Faradaic Current [A/m^2]

i_dl_an = i_ex/A_sg_an - i_far_an # Double Layer current [A/m^2]

## Tracking moles of lithium
mol_an = Delta_y_an*C_Li_an # moles of Lithium in the Anode per unit area [moles/m^2]

'''
Plotting
'''

plt1 = plt.figure(1)
# Current to the Anode
plt.plot(time,i_dl_an,'.-', color='firebrick')
plt.plot(time, i_ex/A_sg_an, linewidth=4, color='deepskyblue')
plt.plot(time,i_far_an, color='black')
#plt.setylim((-2*max(abs(i_external))/A_sg_an ,1.5*max(abs(i_external))/A_sg_an))
plt.title("Current in the Anode")
plt.xlabel("time [s]")
plt.ylabel("Current [A/m^2]")
plt.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5), loc = 'center left')

plt1.tight_layout()

fig2, (ax3, ax4) = plt.subplots(2)
# Double Layer Potential Difference 
ax3.plot(time,Delta_Phi_dl_an)
ax3.set_title(r"$\Delta\Phi_{dl}$")
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Change in Potential [V]")

# Open Cell Potential
ax4.plot(time,U_cell_an)
ax4.set_title("Open Cell Potential")
ax4.set_xlabel("time [s]")
ax4.set_ylabel("Equilibrium Potential [V]")
fig2.tight_layout()

fig3, (ax5, ax6, ax7) = plt.subplots(3)
# Amount of Lithium
ax5.plot(time,mol_an)
ax5.set_title("Amount of Lithium")
ax5.set_xlabel("time [s]")
ax5.set_ylabel("Lithium [mol/m^2]")

# Mole fraction of Lithium
ax6.plot(time,C_Li_an/Anode.C_int[Anode.ind_track])
#ax6.plot(time,np.ones_like(C_Li_an),'r--')
#ax6.plot(time,np.zeros_like(C_Li_an),'r--')
ax6.set_title("Effective Concentration of Lithium")
ax6.set_xlabel("time [s]")
ax6.set_ylabel("Lithium [-]")

# Concentration of Lithium
ax7.plot(time,C_Li_an)
#ax7.plot(time,np.zeros_like(C_Li_an),'r--')
ax7.set_title("Concentration of Lithium")
ax7.set_xlabel("time [s]")
ax7.set_ylabel("Lithium [mol/m^3]")

fig3.tight_layout()

# Overpotential
fig4, (ax8, ax9) = plt.subplots(2)
ax8.plot(time,Delta_Phi_dl_an-U_cell_an)
ax8.plot(time,np.zeros_like(time),':r')
ax8.set_title("Electrode Overpotential")
ax8.set_xlabel("time [s]")
ax8.set_ylabel("Overpotential [V]")

#Exchange Current Density
ax9.plot(time,i_o_an)
ax9.set_title("Exchange Current Density")
ax9.set_xlabel("time [s]")
ax9.set_ylabel(r'$i_o [A/m^2]$')

fig4.tight_layout()

plt.show()
