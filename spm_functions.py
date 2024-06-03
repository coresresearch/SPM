# spm_functions.py
#
#  This file holds utility functions called by the spm model.

import numpy as np 
import math

def Half_Cell_Eqlib_Potential(HalfCell,F = 96.48534, T_amb = 298.15, R = 0.0083145):
    """
    Returns the half cell potential
    """
    # F = 96.48534 #Faraday's number [kC/equivalence]
    # R = 0.0083145 #Universal gas constant [kJ/mol-K]
    n_elc = HalfCell.n
    #T_amb = 273.15 + 25 [K]
    T = HalfCell.Temp
     
    Delta_G_cell = np.dot(HalfCell.G,HalfCell.nu)
    Delta_S = np.dot(HalfCell.S,HalfCell.nu)

    U_0_Cell_amb =  -Delta_G_cell/(n_elc*F)
    U_0_Cell = U_0_Cell_amb + (T- T_amb)*Delta_S/(n_elc*F)
    U_Cell = U_0_Cell - R*T/n_elc/F*np.log(np.prod(np.power(HalfCell.C,HalfCell.nu)))
    return U_Cell

def Butler_Volmer(i_o,V,U,BnF_RT_a,BnF_RT_c):
    """
    This function calculates the faraday current density at the electrode-electrolyte interface, using the
    Butler-Volmer model (A/m2). Positive current is defined as positive current delivered from the electrolyte to the
    electrode.
    
    Parameters
    ----------
    i_o : Exchange Current Density [mA/cm^2] 
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
    i : current density at the electrode-electrolyte interface [mA/cm^2]
    """
    i_far= i_o*(math.exp(-BnF_RT_a*(V-U)) - math.exp(BnF_RT_c*(V-U)))
    return i_far

class Species:
    """
    Defines the thermodynamic properties of the species
    """
    def __init__(self, Name, Gibbs_energy_formation, Standard_Entropy,Standard_State,charge):
        self.name = Name
        self.DG_f = Gibbs_energy_formation
        self.S = Standard_Entropy
        self.state = Standard_State
        self.charge = charge

class Participant(Species):
    """
    Subclass of Species
    Takes in the Species, stoichiometric coefficient, and concentration   
    """
    def __init__(self, Species, stoichiometric_coefficient, concentration):
        super().__init__(Species.name, Species.DG_f, Species.S, Species.state,Species.charge)
        self.stoich_coeff = stoichiometric_coefficient
        self.C = concentration

class Half_Cell:
    """
    Takes in the reactants, products, number of electrons, and the temerature of the half cell in Kelvin
    Stores each property in an array where the reactants are followed by the products
    """
    def __init__(self, Reactants,Products,n,Temperature):
        self.n = n
        self.Temp = Temperature
        indx = 0
        self.name = [None]*(len(Reactants)+len(Products))
        self.G = [None]*(len(Reactants)+len(Products))
        self.S = [None]*(len(Reactants)+len(Products))
        self.C = [None]*(len(Reactants)+len(Products))
        self.nu = [None]*(len(Reactants)+len(Products))
        for i in Reactants:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.C[indx] = i.C
            self.nu[indx] = i.stoich_coeff*-1
            indx = indx + 1
        for i in Products:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.C[indx] = i.C
            self.nu[indx] = i.stoich_coeff
            indx = indx + 1

def residual(_,SV,i_ext,Anode,Cathode,sigma_s,t_s):
    '''
    Derivations (a=anode,s=sperator,c=cathode)

    Change in Double Layer potential Anode [0]:
    Eta_a = Phi_a - Phi_el_a - U_a ; Phi_an = 0 ; Phi_a = Phi_el_a - delta_Phi_dl_a
    Eta_a = - delta_Phi_dl_a - U_a => sub into Butler Volmer
    i_ext/A_sg = i_dl_a + i_far_a ; -i_dl_a = Cap_dl_a*(d Delta_Phi_dl_a/ dt) 
    (i_far_a - i_ext/A_sg)/Cap_dl_a = d Delta_Phi_dl_a/dt

    Change in Lithium concentration in the Anode [1]:
    dN_Li/dt = -s_dot_Li+*A_surf*N_p ; dC_Li/dt = (dN_Li/dt)/(V*N_p) ; s_dot_Li+ = -i_far*nu_Li+/(n*F) ; A_suf/V = A_s
    dC_Li/dt = i_far*nu_Li+*A_s/(n*F)
    
    Change in Double Layer potential Cathode [2]: 
    Eta_c = Phi_c - Phi_el_c - U_c ; Delta_Phi_dl_c = Phi_c - Phi_el_c 
    Eta_c = Delta_Phi_dl_c - U_c => sub into Butler Volmer
    i_ext/A_sg = i_dl_c + i_far_c ; i_dl_c = Cap_dl_c*(d Delta_Phi_c_dl/ dt) 
    (i_far_c - i_ext/A_sg)/Cap_dl_c = d Delta_Phi_c_dl/dt
    '''
    # Anode
    V_a = SV[0]
    C_Li = SV[Anode.indx_Li]
    Anode.HC.C[Anode.indx_Li] = C_Li # Updates the concentraion of Lithium in the anode
    
    U_a = Half_Cell_Eqlib_Potential(Anode.HC)
    i_far_a= Butler_Volmer(Anode.i_o,V_a,U_a,Anode.BnF_RT_a,Anode.BnF_RT_c)
   
    dPhi_dl_a_dt = (i_far_a - i_ext/Anode.A)/Anode.Cap  # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    dC_Li_a_dt = i_far_a*Anode.nuA_nF # returns an expression for dC_Li/dt in terms of Delta_Phi_dl
    
    # Seperator
    # I am not sure if I should set the current in the sperator based off of the change in concentraion of Li+, 
    # or if I should set it equal to the faradaic current leaving the anode, or if it is i_ext
    i_sep = dC_Li_a_dt*Anode.nuA_nF/Anode.A*t_s # using this as a round about way to avoid passing in F
    Delta_Phi_s = -i_sep*t_s/sigma_s # Potential drop across the seperator [V]
    
    # I get a math range error in this part which is why I have it commented out for now
    '''
    # Cathode
    V_c = Delta_Phi_s + SV[2]
    U_c = Half_Cell_Eqlib_Potential(Cathode.HC)
    i_far_c= Butler_Volmer(Cathode.i_o,V_c,U_c,Cathode.BnF_RT_a,Cathode.BnF_RT_c)
    
    dPhi_dl_c_dt = (i_far_c - i_ext/Cathode.A)/Cathode.Cap
    dC_Li_c_dt = i_far_c*Cathode.nuA_nF 
    '''
    dSVdt = [dPhi_dl_a_dt,dC_Li_a_dt] #,dPhi_dl_c_dt,dC_Li_c_dt]
    
    return dSVdt

class Electrode:
    '''
    Holds all of the parameters of an electrode needed in the residual function
    '''
    def __init__(self, BnF_RT_a,BnF_RT_c,Cap,i_o,A,nuA_nF,HC,indx_Li):
        self.BnF_RT_a = BnF_RT_a
        self.BnF_RT_c = BnF_RT_c
        self.Cap = Cap
        self.i_o = i_o
        self.A = A
        self.nuA_nF = nuA_nF
        self.HC = HC
        self.indx_Li = indx_Li