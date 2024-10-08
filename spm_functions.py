# spm_functions.py
#
#  This file holds utility functions called by the spm model.

import numpy as np 
import math
from scipy.optimize import fsolve

def Half_Cell_Eqlib_Potential(HalfCell,F = 96485.34, T_amb = 298.15, R = 8.3145):
    """
    Returns the half cell potential
    
    Parameters
    ----------
    HalfCell : Object contianing the reaction participants and thier properties
    F : Optional, Faraday's number 
        The default is 96485.34 [C/equivalence]
    T_amb : Optional, Standard Temperature  
        The default is 298.15 [K]
    R : Optional, Universal gas constant
        The default is 8.3145 [kJ/mol-K]
    
     Returns
    -------
    U_Cell : The equalibrium (open cell) potential for the half cell [V]
    """
    n_elc = HalfCell.n
    #T_amb = 273.15 + 25 [K]
    print("n",n_elc)
    #T_amb = 273.15 + 25 #[K]
    print("n",n_elc)
    #T_amb = 273.15 + 25 #[K]
    #T_amb = 273.15 + 25 [K]
    T = HalfCell.Temp
     
    Delta_G_cell = np.dot(HalfCell.G,HalfCell.nu) # Standard Gibbs Free Energy for the half cell reaction
    Delta_S = np.dot(HalfCell.S,HalfCell.nu) # Standard Entropy for the half cell reaction

    U_0_Cell_amb =  -Delta_G_cell/(n_elc*F) # Standard half cell equalibrium potential
    U_0_Cell = U_0_Cell_amb + (T- T_amb)*Delta_S/(n_elc*F) # Adjust for temperature
    U_Cell = U_0_Cell - R*T/n_elc/F*np.log(np.prod(np.power(HalfCell.activity,HalfCell.nu))) # adjust for concentration
    return U_Cell

def Butler_Volmer(i_o,V,U,BnF_RT_a,BnF_RT_c):
    """
    This function calculates the faraday current density at the electrode-electrolyte interface, using the
    Butler-Volmer model (A/m2). Positive current is defined as positive current delivered from the electrolyte to the
    electrode.
    
    Parameters
    ----------
    i_o : Exchange Current Density [A/m^2] 
    V : Electrode potential difference at the electrode-electrolyte interface (phi_ed - phi_elyte) [V]
    i_o : Exchange Current Density [mA/cm^2] 
    V : Electrode potential difference at the electrode-electrolyte interface [V]
    U : Equilibrium potential [V]
 
    Inside BnF_RT: a is for anodic, c is for cathodic
    F : Faraday's number
        96485.34 [C/equivalence]
    Beta : The fraction of the total energy that impacts the activation energy of the cathode
        (Typicaly 0.5) [-]
    n: number of electrons per mole of reaction
        [equivalence/mol]
    R : Universal gas constant
        8.3145 [J/mol-K]
    T : Temperature of the interface [K]  
    V : Electrode potential difference at the electrode-electrolyte interface (phi_ed - phi_elyte) [V]
    U : Equilibrium potential [V]
    T : Temperature of the interface [K]    
    F : Optional, Faraday's number
        The default is 96.48534 [kC/equivalence]
    Beta : Optional, The fraction of the total energy that impacts the activation energy of the cathode
        The default is 0.5
    R : Optional, Universal gas constant
        The default is 0.0083145 [kJ/mol-K]
     n: Optional, number of electrons
        The default is 1 [equivalence/mol]
        
    Returns
    -------
    i : current density at the electrode-electrolyte interface [A/m^2]
    """
    i_far= i_o*(math.exp(-BnF_RT_a*(V-U)) - math.exp(BnF_RT_c*(V-U)))
    return i_far

class Species:
    """
    Defines the thermodynamic properties of the species
    """
    def __init__(self, Name, Gibbs_energy_formation, Standard_Entropy,Standard_State,charge):
        self.name = Name
        self.DG_f = Gibbs_energy_formation # [J/mol]
        self.S = Standard_Entropy          # [J/mol-K]
        self.C_int = Standard_State        # [mol/m^3]
        self.charge = charge               # [elementary charge]

class Participant(Species):
    """
    Subclass of Species
    Takes in the Species, stoichiometric coefficient, and concentration
    """
    def __init__(self, Species, stoichiometric_coefficient, concentration, activity_coefficient):
        super().__init__(Species.name, Species.DG_f, Species.S, Species.C_int ,Species.charge)
        self.stoich_coeff = stoichiometric_coefficient # Takes in the magnitude, the sign is added when creating a Half Cell
        self.C = concentration             # [mol/m^3]
        self.gamma = activity_coefficient   # [-]

class Half_Cell:
    """
    Takes in the reactants, products, number of electrons per mol of reaction, and the temerature of the half cell in Kelvin
    Stores each property in an array where the reactants are followed by the products
    
    Activity is the activity coefficient gamma multiplied by the effective concentraion 
    """
    def __init__(self,Reactants,Products,n,Temperature,Beta,F,R,Cap,i_o,A_sg,A_s,Species_Tracked,ion,
                 D_k):
        # Cap: Double layer capacitance
        # D_k: Diffusion coefficient
        self.n = n
        self.Temp = Temperature
        self.BnF_RT_an = Beta*n*F/R/Temperature
        self.BnF_RT_ca = (1-Beta)*n*F/R/Temperature
        self.Cap = Cap
        self.i_o_reff = i_o
        self.A_sg = A_sg
        self.Beta = Beta
        self.D_k = D_k
        
        indx = 0
        self.name = [None]*(len(Reactants)+len(Products))
        self.G = [None]*(len(Reactants)+len(Products))
        self.S = [None]*(len(Reactants)+len(Products))
        self.nu = [None]*(len(Reactants)+len(Products))
        self.C_int = [None]*(len(Reactants)+len(Products))
        self.activity = [None]*(len(Reactants)+len(Products))
        self.gamma = [None]*(len(Reactants)+len(Products))
        for i in Reactants:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.nu[indx] = i.stoich_coeff*-1
            self.C_int[indx] = i.C_int
            self.gamma[indx] = i.gamma
            self.activity[indx] = i.gamma*(i.C/i.C_int)
            indx = indx + 1
        for i in Products:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.nu[indx] = i.stoich_coeff
            self.C_int[indx] = i.C_int
            self.gamma[indx] = i.gamma
            self.activity[indx] = i.gamma*(i.C/i.C_int)
            indx = indx + 1
        # I need to rename these to match the convention of oxidize and reduced, but I am not
        #   positive I have that correct to I am keeping it this way for now. 
        self.ind_track = self.name.index(Species_Tracked)
        self.ind_ion = self.name.index(ion)
        # I no longer force this to be postive. The coeff of th ion is automatically set, but the 
        #   value for n if determed by the user since I do not include the elctron as a species in the reaction
        self.nuA_nF = self.nu[self.ind_ion]*A_s/n/F # [mol_Li+/C-m]

class internal_electrode_geom:
    '''
    Contains the geometry for the radial discritization for the Electrode
    
    Radial nodes are in the middle of each shell volume
    
    The first index is r = 0, index n is r = r_o
    '''
    # I uses equally spaced radial points
    def __init__(self,r_p,n_r,A_sg,Delta_y):
        self.r_p = r_p # radius of the particle 
        self.n_r = n_r # number of radial nodes
        self.dr = r_p/n_r # distance between nodes
        self.A_sg = A_sg # interface surface area per geometric surface area 
        
        self.r_shell = np.arange(n_r+1)*r_p/n_r # radi of each shell [m]
        self.A_shell = 4*np.pi*(self.r_shell**2) # surface area of each shell [m^2]
        self.diff_vol = (4/3)*np.pi*(self.r_shell[1:]**3 - self.r_shell[:-1]**3) # volume between each pair of shells (one volume per node) [m^3]

        self.r_node =  (self.r_shell[:-1] + self.r_shell[1:])/2 # radi of each node [m]
        self.Delta_y = Delta_y # Electrode thickness [m]
        
class Seperator:
    '''
    contians all the sperator information needed for the residual function
    '''
    def __init__(self,D_k,sigma,thickness,n_y,z,F,R,T):
        self.D_k = D_k
        self.sigma = sigma
        self.dy = thickness/(n_y+1)
        self.n_y = n_y
        self.zF_RT = z*F/(R*T)

class Index_start:
    '''
    contains all of the index boundries for the State Variable (SV) vector
    I only track the starts because python indexes in the form {this index}:{one before this index}
    so the starting index of the next section works as the ending index of the previous section
    '''
    def __init__(self,n_r_an,n_r_ca,n_y_sep):
        self.dPhi_an = 0                    
        self.C_an = 1
        self.dPhi_ca = n_r_an + 1
        self.C_ca = n_r_an + 2
        self.C_sep = n_r_an + n_r_ca + 2
        self.Phi_sep = n_r_an + n_r_ca + n_y_sep + 4
           
def radial_molar_flux(Electrode, Geometry, C, s_dot):
    '''
    C: Molar cocentrations at each node [mol/m^3]
    s_dot: species production rate at the surface [mol/m^2-s]
    '''
    N_r_Li = np.zeros(Geometry.n_r + 1)
    
    N_r_Li[0] = 0 # No flux at the center of the electrode
    
    N_r_Li[1:-1] = -Electrode.D_k*(np.subtract(C[1:],C[:-1]))/Geometry.dr

    N_r_Li[-1] = -s_dot # Flux at the surface is equal and opposite to the production rate 
    
    return N_r_Li

def seperator_molar_flux_dae(Phi_sep,seperator,C):
    '''
    the one i use with the dae solver
    '''
    # I track concentrations for the electroltye in each Electrode and at each node inside the seperator
    
    N_y_Li_plus = np.zeros(seperator.n_y + 1)
    
    N_y_Li_plus = -seperator.D_k*((C[1:] - C[:-1])/seperator.dy) - seperator.D_k*seperator.zF_RT*(
        C[1:] + C[:-1])/2*(Phi_sep[1:] - Phi_sep[:-1])/seperator.dy
    
    return N_y_Li_plus

def update_activities(C_electrode,Electrode,C_electrolyte):
    '''
    Updates the activites in the electrode based on the current concentrations then calculates the 
    open cell potential and exchange current density. For now the concentration of the Li+ in the
    electroltye phase is constant so its activity is not updated. The concentration of the outermost
    control volume is used for the activites.
    
    Parameters
    ----------
    C_electrode : Lithium Concentrations in the Electrode [mol/m^3] 
    Electrode : the Electrode that is being updated
    C_electrolyte : Lithium ion Concentrations in the electrolyte phase of the Electrode [mol/m^3] 
    
    Returns
    -------
    Electrode : the Electrode with updated activites
    i_o : exchange current density [A/m^2]
    U : The open cell potential [V]
    '''
    X_Li = C_electrode[-1]/Electrode.C_int[Electrode.ind_track] # effective molar concentration of Lithium at the Electrode surface [-]
    Electrode.activity[Electrode.ind_track] = Electrode.gamma[Electrode.ind_track]*(X_Li) # update activity of the full intercalation compound
    Electrode.activity[-1] = Electrode.gamma[-1]*(1 - X_Li) # update activity of the empty intercalation compound
    Electrode.activity[Electrode.ind_ion] = Electrode.gamma[Electrode.ind_ion]*(C_electrolyte/Electrode.C_int[Electrode.ind_ion]) # update activity of Li+
    # Adjust exchange current density for concentration using a reference concentration
    i_o = ((Electrode.activity[Electrode.ind_track])**Electrode.Beta)*(
        (Electrode.activity[Electrode.ind_ion]*Electrode.activity[-1])**(1-Electrode.Beta))*Electrode.i_o_reff
    
    # Adjust open cell potential for concentration using activity
    U = Half_Cell_Eqlib_Potential(Electrode)
    
    return Electrode, i_o, U

def node_plot_labels(nodes,opening_label):
    '''
    Creates a string to be used in the legend when ploting values for individual nodes. 
    '''
    label = []
    nodes = nodes*1e6 # convert from meters to micrometers
    for ele in nodes:
        label.append(opening_label+str(round(ele,3))+r"$\mu m$")
    return label

def dae_initial_guess(guesses,i_ext,seperator):
    def f(x):
        functions = np.empty(len(x))
        functions[0:-1] = i_ext + seperator.sigma*(x[1:] - x[:-1])/seperator.dy
        functions[-1] = x[0]
        #functions[0] = i_ext + 2*seperator.sigma*(x[0] - 0)/seperator.dy
        #functions[-1] = i_ext + 2*seperator.sigma*(x[-1] - x[-2])/seperator.dy
        return functions
    new_guesses = fsolve(f,guesses)
    return new_guesses 

def residual_dae(t,SV,SV_dot,resid,user_data):
    '''
    Derivations (a=anode,s=sperator,c=cathode)

    Change in Double Layer potential Anode [0]:
    Eta_a = Phi_a - Phi_el_a - U_a ; Phi_an = 0 ; Phi_a - Phi_el_a = delta_Phi_dl_a
    Eta_a = delta_Phi_dl_a - U_a => sub into Butler Volmer
    i_ext/A_sg = i_dl_a + i_far_a ; -i_dl_a = Cap_dl_a*(d Delta_Phi_dl_a/ dt) 
    (-i_far_a + i_ext/A_sg)/Cap_dl_a = d Delta_Phi_dl_a/dt

    Change in Lithium concentration in the Anode [1]:
    dN_Li/dt = -s_dot_Li+*A_surf*N_p ; dC_Li/dt = (dN_Li/dt)/(Vol*N_p) ; s_dot_Li+ = -i_far*nu_Li+/(n*F) ; A_suf/Vol = A_s
    dC_Li/dt = i_far*nu_Li+*A_s/(n*F)
    
    Change in Double Layer potential Cathode [2]: 
    Eta_c = Phi_c - Phi_el_c - U_c ; Phi_c - Phi_el_c = delta_Phi_dl_c 
    Eta_c = delta_Phi_dl_a - U_c => sub into Butler Volmer
    -i_ext/A_sg = i_dl_c + i_far_c ; i_dl_c = Cap_dl_c*(d Delta_Phi_c_dl/ dt) 
    (-i_far_c - i_ext/A_sg)/Cap_dl_c = d Delta_Phi_c_dl/dt
    
    Change in Lithium concentration in the Cathode [3]:
    dC_Li/dt = i_far*nu_Li+*A_s/(n*F)
    '''
    # The convention for double layer potential is electrode minus electrolyte
    # For this reaction, Li+ and the electron are both on the same side of the reaction,
    #   so the signs of their coefficients will always be the same which means nuA_nF
    #   is positive. As a result, a positive i_far will cause the concentration
    #   of lithium in the electrode to increase
    # I hard code in the positions for the empty electrode in this reaction. Since I use the reaction
    #   Lithiated_Electrode -> Li+ + Electrode + e- for both half cells, the empty elcetrode is the last partcipant in 
    #   the list in both HCs
    # I adjust the exchange current densities for concentration 
    # I use mole fractions in place of activites when adjusting for concentration with the Open Cell Potential
    
    i_ext = user_data[0]
    Anode = user_data[1]
    Geom_an = user_data[2]
    Cathode = user_data[3]
    Geom_ca = user_data[4]
    seperator = user_data[5]
        
    ## Anode
    C_electrolyte_an = SV[Geom_an.n_r+Geom_ca.n_r+2]
    Phi_dl_an = SV[0]
    C_an =  SV[1:Geom_an.n_r+1] # Lithium concentration in the anode
    
    Anode, i_o_an, U_a = update_activities(C_an,Anode,C_electrolyte_an)

    i_far_a= Butler_Volmer(i_o_an,Phi_dl_an,U_a,Anode.BnF_RT_an,Anode.BnF_RT_ca)
    i_dl_a = i_ext/Geom_an.A_sg - i_far_a # Double Layer current
   
    #dPhi_dl_a_dt = (-i_far_a + i_ext/Anode.A_sg)/Anode.Cap  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    resid[0] = SV_dot[0] - (-i_far_a + i_ext/Anode.A_sg)/Anode.Cap
    
    s_dot_far_an = i_far_a*Anode.nuA_nF/(3/Geom_an.r_p) # Li species production rate at the surface as a result of i_far
    s_dot_dl_an = i_dl_a*Anode.nuA_nF/(3/Geom_an.r_p) # Li plus species movement rate from the elctroltye to the dl as a result of i_dl
    
    N_r_Li_an =  radial_molar_flux(Anode, Geom_an, C_an, s_dot_far_an)
    # Flux in minus flux out (closer to center minus closer to surface)
    dN_r_Li_an_dt = np.subtract(np.transpose(N_r_Li_an[:-1])*Geom_an.A_shell[:-1] , np.transpose(N_r_Li_an[1:])*Geom_an.A_shell[1:])
    # Divide by the volume to the get the concentration rate
    dC_Li_a_dt = np.transpose(dN_r_Li_an_dt)/Geom_an.diff_vol
    #dSVdt[1:Geom_an.n_r+1] = SV_dot[1:Geom_an.n_r+1] - np.transpose(dN_r_Li_an_dt)/Geom_an.diff_vol
    for ind in range(1,Geom_an.n_r+1, 1):
        resid[ind] = SV_dot[ind] - dC_Li_a_dt[ind-1]
        
    ## Cathode
    Phi_dl_ca = SV[Geom_an.n_r+1]
    C_electrolyte_ca = SV[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+3]
    C_ca =  SV[Geom_an.n_r+2:Geom_an.n_r+Geom_ca.n_r+2] # Lithium concentration in the anode

    Cathode, i_o_ca, U_c = update_activities(C_ca,Cathode,C_electrolyte_ca)

    i_far_c = Butler_Volmer(i_o_ca,Phi_dl_ca,U_c,Cathode.BnF_RT_an,Cathode.BnF_RT_ca)
    i_dl_c = -i_ext/Geom_ca.A_sg - i_far_c # Double Layer current
    
    #dPhi_dl_c_dt = (-i_far_c - i_ext/Cathode.A_sg)/Cathode.Cap
    resid[Geom_an.n_r+1] = SV_dot[Geom_an.n_r+1] - (-i_far_c - i_ext/Cathode.A_sg)/Cathode.Cap
    
    # nu in these next two lines is for Li+
    s_dot_far_ca = i_far_c*Cathode.nuA_nF/(3/Geom_ca.r_p) # species production rate at the surface as a result of i_far [mol/m^2]
    s_dot_dl_ca = i_dl_c*Cathode.nuA_nF/(3/Geom_ca.r_p) # species movement rate from the elctroltye to the dl as a result of i_dl [mol/m^2]

    N_r_Li_ca =  radial_molar_flux(Cathode, Geom_ca, C_ca, s_dot_far_ca)
    # Flux in minus flux out (closer to center minus closer to surface)
    dN_r_Li_ca_dt = np.subtract(np.transpose(N_r_Li_ca[:-1])*Geom_ca.A_shell[:-1] , np.transpose(N_r_Li_ca[1:])*Geom_ca.A_shell[1:])
    # Divide by the volume to the get the concentration rate
    dC_Li_c_dt = np.transpose(dN_r_Li_ca_dt)/Geom_ca.diff_vol
    #dSVdt[Geom_an.n_r+2:Geom_an.n_r+Geom_ca.n_r+2] = SV_dot[Geom_an.n_r+2:Geom_an.n_r+Geom_ca.n_r+2] - np.transpose(dN_r_Li_ca_dt)/Geom_ca.diff_vol
    for ind in range(Geom_an.n_r+2,Geom_an.n_r+Geom_ca.n_r+2, 1):
        resid[ind] = SV_dot[ind] - dC_Li_c_dt[ind-(Geom_an.n_r+2)]
        
    ## Seperator
    C_sep = SV[Geom_an.n_r+Geom_ca.n_r+2:Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4] # Lithium ion concentration in the eletrolyte and seperator
    Phi_sep = SV[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4:]

    N_y_Li_plus = seperator_molar_flux_dae(Phi_sep,seperator,C_sep)
    
    resid[Geom_an.n_r+Geom_ca.n_r+3:Geom_an.n_r+Geom_ca.n_r+seperator.n_y+3] = SV_dot[Geom_an.n_r+Geom_ca.n_r+3:Geom_an.n_r+Geom_ca.n_r+seperator.n_y+3] - (N_y_Li_plus[:-1] - N_y_Li_plus[1:])/seperator.dy  # in minus out
    
    # concentration in the electrolyte in the anode
    resid[Geom_an.n_r+Geom_ca.n_r+2] = SV_dot[Geom_an.n_r+Geom_ca.n_r+2] - (s_dot_far_an*Geom_an.A_sg + s_dot_dl_an*Geom_an.A_sg - N_y_Li_plus[0])/(seperator.dy/2)
    
    # concentration in the electrolyte in the cathode   
    resid[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+3] = SV_dot[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+3] - (N_y_Li_plus[-1] + s_dot_far_ca*Geom_ca.A_sg + s_dot_dl_ca*Geom_ca.A_sg)/(seperator.dy/2)
    
    ####### This is the line to uncomment/commnet out if I do/do not want the seperator running
    #resid[Geom_an.n_r+Geom_ca.n_r+2:Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4] = SV_dot[Geom_an.n_r+Geom_ca.n_r+2:Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4] # no change in the seperator concentrations
    
    # Phi of the seperator (relative to the first node)
    resid[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4:-1] = i_ext + seperator.sigma*(SV[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+5:] - SV[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4:-1])/seperator.dy
    resid[-1] = SV[Geom_an.n_r+Geom_ca.n_r+seperator.n_y+4] # sets the first node to have a potential of zero

