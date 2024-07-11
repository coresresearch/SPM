# spm_functions.py
#
#  This file holds utility functions called by the spm model.

import numpy as np 
import math

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
        self.gamma =activity_coefficient   # [-]

class Half_Cell:
    """
    Takes in the reactants, products, number of electrons per mol of reaction, and the temerature of the half cell in Kelvin
    Stores each property in an array where the reactants are followed by the products
    
    Activity is the activity coefficient gamma multiplied by the effective concentraion 
    """
    def __init__(self,Reactants,Products,n,Temperature,Beta,F,R,Cap,i_o,A_sg,A_s,Species_Tracked,ion):
        self.n = n
        self.Temp = Temperature
        self.BnF_RT_an = Beta*n*F/R/Temperature
        self.BnF_RT_ca = (1-Beta)*n*F/R/Temperature
        self.Cap = Cap
        self.i_o_reff = i_o
        self.A_sg = A_sg
        self.Beta = Beta
        
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
        self.nuA_nF = self.nu[self.ind_ion]*A_s/n/F

def residual(_,SV,i_ext,Anode,Cathode):
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
     
    # Anode
    V_a = SV[0]
    
    X_Li_a = SV[1]/Anode.C_int[Anode.ind_track] # effective molar concentration of Lithium (LiC6) in the anode [-]
    Anode.activity[Anode.ind_track] = Anode.gamma[Anode.ind_track]*(X_Li_a) # update activity of the LiC6
    Anode.activity[-1] = Anode.gamma[-1]*(1 - X_Li_a) # update activity of the C6
    # The activity of the Li+ in the electroltye does not change because I am assuming the concentration is constant
    
    # Adjust exchange current density for concentration
    #   uses effective molar concentration which is activity/activity_coefficient 
    i_o_an = ((X_Li_a)**Anode.Beta)*(
        (Anode.activity[Anode.ind_ion]/Anode.gamma[Anode.ind_ion])**(1-Anode.Beta))*Anode.i_o_reff
    
    U_a = Half_Cell_Eqlib_Potential(Anode)
    i_far_a= Butler_Volmer(i_o_an,V_a,U_a,Anode.BnF_RT_an,Anode.BnF_RT_ca)
   
    dPhi_dl_a_dt = (-i_far_a + i_ext/Anode.A_sg)/Anode.Cap  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    dC_Li_a_dt = i_far_a*Anode.nuA_nF # returns an expression for dC_Li/dt in terms of Delta_Phi_dl
    
    # Cathode
    V_c = SV[2]
    
    X_Li_c = SV[3]/Cathode.C_int[Cathode.ind_track] # effective molar concentration of Lithium (LiFePO4) in the cathode [-]
    Cathode.activity[Cathode.ind_track] = Cathode.gamma[Cathode.ind_track]*(X_Li_c) # update activity of the LiFePO4
    Cathode.activity[-1] = Cathode.gamma[-1]*(1 - X_Li_c) # update activity of the FePO4
    
    # Adjust exchange current density for concentration 
    #   uses effective molar concentration which is also the activity/activity_coefficient 
    i_o_ca = ((X_Li_c)**Cathode.Beta)*(
        (Cathode.activity[Cathode.ind_ion]/Cathode.gamma[Cathode.ind_ion])**(1-Cathode.Beta))*Cathode.i_o_reff
    
    U_c = Half_Cell_Eqlib_Potential(Cathode)
    i_far_c = Butler_Volmer(i_o_ca,V_c,U_c,Cathode.BnF_RT_an,Cathode.BnF_RT_ca)
    
    dPhi_dl_c_dt = (-i_far_c - i_ext/Cathode.A_sg)/Cathode.Cap
    dC_Li_c_dt = i_far_c*Cathode.nuA_nF

    dSVdt = [dPhi_dl_a_dt, dC_Li_a_dt, dPhi_dl_c_dt, dC_Li_c_dt]
    
    return dSVdt

def residual_single(_,SV,i_ext,Anode):
    '''
    Same as the other residual function, but pared down for one electrode 
    '''
    # Anode
    V_a = SV[0]
    
    X_Li_a = SV[1]/Anode.C_int[Anode.ind_track] # effective molar concentration of Lithium (LiC6) in the anode [-]
    Anode.activity[Anode.ind_track] = Anode.gamma[Anode.ind_track]*(X_Li_a) # update activity of the LiC6
    Anode.activity[-1] = Anode.gamma[-1]*(1 - X_Li_a) # update activity of the C6
    # The activity of the Li+ in the electroltye does not change because I am assuming the concentration is constant
    
    # Adjust exchange current density for concentration
    i_o_an = ((X_Li_a)**Anode.Beta)*(
        (Anode.activity[Anode.ind_ion]/Anode.gamma[Anode.ind_ion])**(1-Anode.Beta))*Anode.i_o_reff
    
    U_a = Half_Cell_Eqlib_Potential(Anode)
    i_far_a= Butler_Volmer(i_o_an,V_a,U_a,Anode.BnF_RT_an,Anode.BnF_RT_ca)
   
    dPhi_dl_a_dt = (-i_far_a + i_ext/Anode.A_sg)/Anode.Cap  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    dC_Li_a_dt = i_far_a*Anode.nuA_nF # returns an expression for dC_Li/dt in terms of Delta_Phi_dl
    
    dSVdt = [dPhi_dl_a_dt, dC_Li_a_dt]
    
    return dSVdt