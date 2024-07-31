# spm.py
#
# This file serves as the main model file.  
#   It is called by the user to run the model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from spm_functions import Butler_Volmer as faradaic_current
from spm_functions import Half_Cell_Eqlib_Potential, residual, update_activities, node_plot_labels
from spm_functions import Species, Participant, Half_Cell, internal_electrode_geom, Seperator

# Anode is on the left at x=0 and Cathode is on the right
# LiC6 -> Li+ + C6 + e- (reaction at the anode)
# LiFePO4 -> FePO4 + Li+ + e- (reaction at the cathode)
# I do not track the movement of Li through the anode, only the rate of creation of Li+ at the surface
# The rate Lithium ions enter the electrolyte from the anode is equal to the rate of Li+ enter the seperator and eventually the cathode
# I assume the concentration in the electrolyte is in uniform so there is no diffusion and there is no bulk movement of the fluid
#   so there is no convection therefore there is only migration which is driven by the potential difference
# Potential drop across the electroltye obeys Ohm's Law
# I do not create an instance of the species/particiant classes for the electron so I track the 
#   sign of the electron speperately using 'n'
# Galvanostatic
# A positive current dose work so it is when the battery. This is when both reactions proceed spontaniously to reduce their 
#   thermodynamic potenitals. This happens durring discharging when a postive external current enters the anode
#   and Lithium ions go from the anode to the cathode.
# I use different diffusion coefficients for each phase

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
i_external = np.array([0,-1000]) # external current into the Anode [A/m^2] 
t_sim_max = [.01,40] # the maximum time the battery will be held at each current [s]
T = 298.15 # standard temperature [K]

# Initial Conditions
Phi_dl_0_an = -0.64 # initial value for Phi_dl for the Anode [V]
X_Li_0_an = 0.35 # Initial Mole Fraction of the Anode for Lithium [-]
Phi_dl_0_ca = 0.34 #0.321 # initial value for Phi_dl for the Cathode [V]
X_Li_0_ca = 0.6 # Initial Mole Fraction of the Cathode for Lithium [-]

# Material parameters:
MW_g = 12 # Molectular Weight of Graphite [g/mol]
rho_g = 2.2e6 # Denstity of graphite [g/m^3]
MW_FePO4 = 150.815 # Molectular Weight of iron phosphate [g/mol]
rho_FePO4 = 2.87e6 # Denstity of iron phosphate [g/m^3]
C_Li_plus = 1000 # Concentration of Li+ in the Electrolyte [mol/m^3] (I assume electrolyte transport is fast)
C_std = 1000 # Standard Concentration [mol/m^3] (same as 1 M)
sigma_sep = 1.2 # Ionic conductivity for the seperator [1/m-ohm] (this is concentration dependent but it is constant for now)

# Activity Coefficients (All 1 for now)
gamma_LiC6 = 1
gamma_C6 = 1
gamma_Li_plus = 1
gamma_LiFePO4 = 1
gamma_FePO4 = 1

# Kinetic parameters
i_o_reff = 120 # Exchange Current Density [A/m^2] 
Cap_dl = 6*10**-5 # Double Layer Capacitance [F/m^2]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
# both Li+ and electrons are products in this reaction so they have postive coefficients
nu_Li_plus = 1 # stoichiometric coefficient of Lithium [mol_Li+/mol_rxn]
n = -1 # number of electrons multiplied by the charge of an electron [mol_electrons/mol_rxn]
D_k_an =  7.5e-13 #Diffusion coefficient of Li in the anode [m^2/s]
D_k_ca =  7.5e-13 #Diffusion coefficient of Li in the cathode [m^2/s]
D_k_sep = 7e-8 #Difusion coefficient of Li+ in the seperator [m^2/s]

# Microstructure
t_sep =  1e-5 # thickness of the seperator [m]
n_y_sep = 4 # number of nodes inside the seperator
Delta_y_an = 50*10**-6 # Anode thickness [m]
r_an = 3*10**-6 # Anode particle radius [m]
n_r_an = 5 # Number of radial nodes in the anode
Delta_y_ca = 50*10**-6 # Cathode thickness [m]
r_ca = 5*10**-6 # Cathode particle radius [m]
n_r_ca = 5 # Number of radial nodes in the cathode
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

LiC6_rxn_an = Participant(LiC6,1,C_Li_0_an,gamma_LiC6)
C6_rxn_an = Participant(C6,1,C_g_0_an,gamma_C6)
Li_plus_rxn_an = Participant(Li_plus,1,C_Li_plus,gamma_Li_plus)

React_an = [LiC6_rxn_an]
Prod_an = [Li_plus_rxn_an,C6_rxn_an]

# Both electrodes have the same reation and in this forward reaction one electron is 
#   produced so n = 1 (as it is defined above in the inputs)
Anode = Half_Cell(React_an,Prod_an,n,T,Beta,F,R,Cap_dl,i_o_reff,A_sg_an,A_s_an,'LiC6','Li+',D_k_an)
Geom_an = internal_electrode_geom(r_an,n_r_an,A_sg_an)

# Cathode reacation: LiFePO4 -> FePO4 + Li+ + e-
LiFePO4 = Species("LiFePO4",-326650,130.95,C_FePO4,0)
FePO4 = Species("FePO4",0,171.3,C_FePO4,0)

LiFePO4_rxn_ca = Participant(LiFePO4,1,C_Li_0_ca,gamma_LiFePO4)
FePO4_rxn_ca = Participant(FePO4,1,C_FePO4_0,gamma_FePO4)
Li_plus_rxn_ca = Participant(Li_plus,1,C_Li_plus,gamma_Li_plus)

React_ca = [LiFePO4_rxn_ca]
Prod_ca = [Li_plus_rxn_ca,FePO4_rxn_ca]

Cathode = Half_Cell(React_ca,Prod_ca,n,T,Beta,F,R,Cap_dl,i_o_reff,A_sg_ca,A_s_ca,'LiFePO4','Li+',D_k_ca)
Geom_ca = internal_electrode_geom(r_ca,n_r_ca,A_sg_ca)

## Set up Seperator
z_Li_plus = nu_Li_plus*1 # charge of Li+
seperator = Seperator(D_k_sep,sigma_sep,t_sep,n_y_sep,z_Li_plus,F,R,T)

'''
Integration
'''
# Integration Limits
def min_voltage(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    V_cell = SV[Geom_an.n_r+1] - i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_min
min_voltage.terminal = True

def max_voltage(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    V_cell = SV[Geom_an.n_r+1] - i_ext*t_sep/sigma_sep - SV[0]
    return V_cell - V_max
max_voltage.terminal = True

def min_Li_an(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    C_Li_check = SV[Geom_an.n_r] # Concentration of Lithium at the surface of the Anode
    return C_Li_check/Anode.C_int[Anode.ind_track] - X_Li_min
min_Li_an.terminal = True

def max_Li_an(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    C_Li_check = SV[Geom_an.n_r] # Concentration of Lithium at the surface of the Anode
    return C_Li_check/Anode.C_int[Anode.ind_track] - X_Li_max
max_Li_an.terminal = True

def min_Li_ca(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    C_Li_check = SV[Geom_an.n_r+Geom_ca.n_r+1] # Concentration of Lithium at the surface of the Cathode
    return C_Li_check/Cathode.C_int[Cathode.ind_track] - X_Li_min
min_Li_ca.terminal = True

def max_Li_ca(_,SV,i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator):
    C_Li_check = SV[Geom_an.n_r+Geom_ca.n_r+1] # Concentration of Lithium at the surface of the Cathode
    return C_Li_check/Cathode.C_int[Cathode.ind_track] - X_Li_max
max_Li_ca.terminal = True
    
sim_inputs = np.zeros(1 + Geom_an.n_r + 1 + Geom_ca.n_r + seperator.n_y + 2 + 1) # [phi_dl, concentrations, phi_dl, concentrations, concentrations, time]

# Start Electrode with uniform concentration
sim_inputs[0] = Phi_dl_0_an                                                             # Initial Phi double layer - anode
sim_inputs[1:Geom_an.n_r+1] = np.ones(Geom_an.n_r)*C_Li_0_an                            # Initial Li concentrations - anode
sim_inputs[Geom_an.n_r+1] = Phi_dl_0_ca                                                 # Initial Phi double layer - cathode
sim_inputs[Geom_an.n_r+2:Geom_an.n_r+Geom_ca.n_r+2] = np.ones(Geom_ca.n_r)*C_Li_0_ca    # Initial Li concentrations - cathode
sim_inputs[Geom_an.n_r+Geom_ca.n_r+2:-1] = np.ones(seperator.n_y + 2)*C_Li_plus         # Initial Li+ concentration - seperator
sim_inputs[-1] = 0                                                                      # Initial time

for ind, i_ext in enumerate(i_external):
 
    # Integration parameters 
    t_start = sim_inputs[-1] # [s]
    t_end = t_start + t_sim_max[ind] # max length of time passed in the integration [s]
    t_span = [t_start,t_end]
    SV_0 = sim_inputs[0:-1] # initial values

    # Find the concentrations in the seperator at steady state and update the activities 
    trans_Li = 0.5  # transference number of Lithium ion 
    C_Li_plus_0_an = C_Li_plus + (t_sep/2 - 0)*(i_ext*(1-trans_Li)/(F*D_k_sep)) # concentration of Li+ in the anode
    Anode.activity[Anode.ind_ion] = Anode.gamma[Anode.ind_ion]*(C_Li_plus_0_an/Anode.C_int[Anode.ind_ion])
    C_Li_plus_0_ca = C_Li_plus + (t_sep/2 - t_sep)*(i_ext*(1-trans_Li)/(F*D_k_sep)) # concentration of Li+ in the anode
    Cathode.activity[Cathode.ind_ion] = Cathode.gamma[Cathode.ind_ion]*(C_Li_plus_0_ca/Cathode.C_int[Cathode.ind_ion])
    
    # Integrater
    SV = solve_ivp(residual,t_span,SV_0,method='BDF',
                args=(i_ext,Anode,Geom_an,Cathode,Geom_ca,seperator),
                rtol = 1e-8,atol = 1e-10, events=(min_voltage, max_voltage, min_Li_an, max_Li_an, min_Li_ca, max_Li_ca))
    
    new_outputs = np.stack((*SV.y, SV.t, i_ext*np.ones_like(SV.t)))
    if ind == 0:
        sim_outputs = new_outputs
    else:
        sim_outputs = np.concatenate((sim_outputs,new_outputs), axis = 1)

    # Once an event is triggered the intergration stops, I want it to just swtich to the next current
    #   so I move the inputs just under the event threshold (back one index) so the integration can proceed 
    for ind, i in enumerate(sim_outputs):
        if ind<len(sim_outputs)-1:
            if SV.status == 1:
                sim_inputs[ind] = i[-2]
            else:
                sim_inputs[ind] = i[-1]
    
'''
Post Processing
'''
Delta_Phi_dl_an = sim_outputs[0]
C_Li_an = sim_outputs[1:Geom_an.n_r+1]
Delta_Phi_dl_ca = sim_outputs[Geom_an.n_r+1]
C_Li_ca = sim_outputs[Geom_an.n_r+2:Geom_an.n_r+Geom_ca.n_r+2]
C_Li_plus_sep = sim_outputs[Geom_an.n_r+Geom_ca.n_r+2:-2]
time = sim_outputs[-2]
i_ex = sim_outputs[-1]

## Anode LiC6 -> Li+ + C6 + e- 
U_cell_an = np.zeros_like(sim_outputs[0])
i_far_an = np.zeros_like(sim_outputs[0])
i_o_an = np.zeros_like(sim_outputs[0])
for ind, ele in enumerate(Delta_Phi_dl_an):
    # I find open cell potential and faradic current using the same process as the residual function
    Anode, i_o_an[ind], U_cell_an[ind] = update_activities(C_Li_an[:,ind],Anode)
    i_far_an[ind] = faradaic_current(i_o_an[ind],Delta_Phi_dl_an[ind],U_cell_an[ind],Anode.BnF_RT_an,Anode.BnF_RT_ca) # Faradaic Current [A/m^2]

i_dl_an = i_ex/A_sg_an - i_far_an # Double Layer current [A/m^2]

## Cathode LiFePO4 -> FePO4 + Li+ + e-
U_cell_ca = np.zeros_like(sim_outputs[0])
i_far_ca = np.zeros_like(sim_outputs[0])
i_o_ca = np.zeros_like(sim_outputs[0])
for ind, ele in enumerate(Delta_Phi_dl_ca):
    Cathode, i_o_ca[ind], U_cell_ca[ind] = update_activities(C_Li_ca[:,ind],Cathode)
    i_far_ca[ind] = faradaic_current(i_o_ca[ind],Delta_Phi_dl_ca[ind],U_cell_ca[ind],Cathode.BnF_RT_an,Cathode.BnF_RT_ca) # Faradaic Current [A/m^2]
    
i_dl_ca = -i_ex/A_sg_ca - i_far_ca # Double Layer current [A/m^2]

## Seperator (Delta_Phi_sep = Phi_el_ca - Ph_el_an)
# A positive external current to the anode should lead ot Li+ ions traveling from the anode 
#   to the cathode. In this case, there should be a potential drop due to the ohmic resistance
#   which would make Phi_el_ca < Ph_el_an and therefore Delta_Phi_sep < 0.
# Long way of saying Delta_Phi_sep should have the opposite sign as i_ext
Delta_Phi_sep = -i_ex*t_sep/sigma_sep # Potential drop across the seperator [V]

## Whole Cell
# Phi_ca - Phi_el_ca = Delta_Phi_dl_ca  
# Phi_an - Phi_el_an = Delta_Phi_dl_an 

# V_cell = Phi_ca - Phi_an 
# V_cell = Ph_ca - Phi_elc_ca + Phi_elc_ca - Phi_elc_an + Phi_elc_an - Phi_an
# V_cell = (Ph_ca - Phi_elc_ca) + (Phi_elc_ca - Phi_elc_an) - (Phi_an - Phi_el_an)

V_cell = Delta_Phi_dl_ca + Delta_Phi_sep - Delta_Phi_dl_an

## Tracking moles of lithium
# I am tracking moles of Li in the electrodes spereately from the electroltye and seperator because
#   those would be per unit area and I am not sure how to match units
mol_an_cv = []
mol_ca_cv = []
for ind, ele in enumerate(C_Li_an):
    mol_an_cv.append(ele*Geom_an.diff_vol[ind]) # moles of Lithium in the Anode particle at each node [mole]
for ind, ele in enumerate(C_Li_ca):
    mol_ca_cv.append(ele*Geom_ca.diff_vol[ind]) # moles of Lithium in the Cathode particle at each node [mole]
mol_an_p = np.sum(mol_an_cv,axis=0) # total moles of Lithium in each particle in the Anode [mole/particle]
mol_ca_p = np.sum(mol_ca_cv,axis=0) # total moles of Lithium in each particle in the Cathode [mole/particle]
mol_an = mol_an_p*Epsilon_g*Delta_y_an/Vol_an # total moles per cross sectional area in the Anode [mole/m^2]
mol_ca = mol_ca_p*Epsilon_g*Delta_y_ca/Vol_ca # total moles per cross sectional area in the Cathode [mole/m^2]

mol_total_electrodes = mol_ca + mol_an  # total moles of Lithium per cross sectional area [mole/m^2]


sep_cv_thickness = np.ones(seperator.n_y + 2)*seperator.dy
sep_cv_thickness[0] = sep_cv_thickness[0]/2 # half nodes at the two ends
sep_cv_thickness[-1] = sep_cv_thickness[-1]/2

mol_sep_cv = []
for ind, ele in enumerate(C_Li_plus_sep):
    mol_sep_cv.append(ele*sep_cv_thickness[ind]) # moles of Lithium in the Seperator per unit area [moles/m^2]
mol_sep = np.sum(mol_sep_cv,axis=0) # total moles of Lithium ions in the Cathode [moles/m^2]


'''
Plotting
'''
# Create a label for each node
r_label_an = node_plot_labels(Geom_an.r_node,r"$r_{an} = $")
    
r_label_ca = node_plot_labels(Geom_ca.r_node,r"$r_{ca} = $")
    
nodes_sep = np.arange(seperator.n_y+2)*round(seperator.dy,2)
y_label = node_plot_labels(nodes_sep,r"$y = $")

fig1, (ax1, ax2) = plt.subplots(2)
# Current to the Anode
ax1.plot(time,i_dl_an,'.-', color='firebrick')
ax1.plot(time, i_ex/A_sg_an, linewidth=4, color='deepskyblue')
ax1.plot(time,i_far_an, color='black')
#ax1.set_ylim((-2*max(abs(i_external))/A_sg_an ,1.5*max(abs(i_external))/A_sg_an))
ax1.set_title("Current in the Anode")
ax1.set_xlabel("time [s]")
ax1.set_ylabel(r"Current $[\frac{A}{m^2}]$")
ax1.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5), loc = 'center left')

# Current to the Cathode
ax2.plot(time,i_dl_ca,'.-', color='firebrick')
ax2.plot(time, i_ex/A_sg_ca, linewidth=4, color='deepskyblue')
ax2.plot(time,i_far_ca, color='black')
#ax2.set_ylim((-2*max(abs(i_external))/A_sg_an ,1.5*max(abs(i_external))/A_sg_an))
ax2.set_title("Current in the Cathode")
ax2.set_xlabel("time [s]")
ax2.set_ylabel(r"Current $[\frac{A}{m^2}]$")
ax2.legend(['DL','EXT','FAR'], bbox_to_anchor=(1, 0.5),loc = 'center left')
fig1.tight_layout()

fig2, (ax3, ax4) = plt.subplots(2)
# Double Layer Potential Difference 
ax3.plot(time,Delta_Phi_dl_an)
ax3.plot(time,Delta_Phi_dl_ca)
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

fig3, (ax5, ax6, ax7) = plt.subplots(3)
# Amount of Lithium
ax5.plot(time,mol_an)
ax5.plot(time,mol_ca)
ax5.plot(time,mol_total_electrodes,'--')
ax5.legend(['Anode','Cathode','Total'], bbox_to_anchor=(1, 0.5),loc = 'center left')
ax5.set_title("Amount of Lithium in the Elcectrode")
ax5.set_xlabel("time [s]")
ax5.set_ylabel(r"Lithium $[\frac{mol}{m^2}]$")

# Mole fraction of Lithium
for ind, ele in enumerate(C_Li_an):
    ax6.plot(time,ele/Anode.C_int[Anode.ind_track],label=r_label_an[ind])
for ind, ele in enumerate(C_Li_ca):
    ax6.plot(time,ele/Cathode.C_int[Cathode.ind_track],label=r_label_ca[ind])
ax6.plot(time,np.ones_like(C_Li_ca[0]),'r--')
ax6.plot(time,np.zeros_like(C_Li_ca[0]),'r--')
ax6.legend(ncol=2, bbox_to_anchor=(1, 0.5),loc = 'center left')
ax6.set_title("Effective Concentration of Lithium")
ax6.set_xlabel("time [s]")
ax6.set_ylabel("Lithium [-]")

# Concentration of Lithium
for ind, ele in enumerate(C_Li_an):
    ax7.plot(time,ele,label=r_label_an[ind])
for ind, ele in enumerate(C_Li_ca):
    ax7.plot(time,ele,label=r_label_ca[ind])
ax7.plot(time,np.zeros_like(C_Li_ca[0]),'r--')
ax7.legend(ncol=2, bbox_to_anchor=(1, 0.5),loc = 'center left')
ax7.set_title("Concentration of Lithium")
ax7.set_xlabel("time [s]")
ax7.set_ylabel(r"Lithium $[\frac{mol}{m^3}]$")

fig3.tight_layout()

# Cell Voltage
plt4 = plt.figure(4)
plt.plot(time,Delta_Phi_dl_an,'--')
plt.plot(time,Delta_Phi_dl_ca,'--')
plt.plot(time,Delta_Phi_sep,'--')
plt.plot(time,V_cell)
plt.legend([r'$\Delta\Phi_{dl,a}$',r'$\Delta\Phi_{dl,c}$',r'$\Delta\Phi_{sep}$', r'$V_{cell}$'], bbox_to_anchor=(1, 0.5),loc = 'center left')
plt.title("Cell Voltage")
plt.xlabel("time [s]")
plt.ylabel("Voltage [V]")
plt.tight_layout()

# Overpotential
fig5, (ax8, ax9) = plt.subplots(2)
ax8.plot(time,Delta_Phi_dl_an-U_cell_an)
ax8.plot(time,(Delta_Phi_dl_ca-U_cell_ca))
ax8.plot(time,np.zeros_like(time),':r')
ax8.legend(['Anode','Cathode'], bbox_to_anchor=(1, 0.5),loc = 'center left')
ax8.set_title("Electrode Overpotential")
ax8.set_xlabel("time [s]")
ax8.set_ylabel("Overpotential [V]")

#Exchange Current Density
ax9.plot(time,i_o_an)
ax9.plot(time,i_o_ca)
ax9.plot(time,np.zeros_like(time),':r')
ax9.legend(['Anode','Cathode'], bbox_to_anchor=(1, 0.5),loc = 'center left')
ax9.set_title("Exchange Current Density")
ax9.set_xlabel("time [s]")
ax9.set_ylabel(r"$i_o\: [A/m^2]$")

fig5.tight_layout()

'''
#Seperator plot on hold for now

#Seperator Concentration
plt6 = plt.figure(6)
for ind, ele in enumerate(C_Li_plus_sep):
    plt.plot(time,ele/Cathode.C_int[Cathode.ind_ion],label=y_label[ind])
plt.legend(ncol=2, bbox_to_anchor=(1, 0.5),loc = 'center left')
plt.title("Effective Concentration of Li+ in the Seperator")
plt.xlabel("time [s]")
plt.ylabel(r"Li+ [-]$")
plt.tight_layout()

plt7 = plt.figure(7)
plt.plot(time,mol_sep)
plt.title("Amount of Lithium in the Seperator")
plt.xlabel("time [s]")
plt.ylabel(r"Li+ $[\frac{mol}{m^2}]$")


plt8 = plt.figure(8)
plt.plot(time,i_far_an*A_sg_an)
plt.plot(time,-i_far_ca*A_sg_ca)
'''

plt.show()