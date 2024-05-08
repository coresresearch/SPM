# spm_functions.py
#
#  This file holds utility functions called by the spm model.

import numpy as np 
import math

def Half_Cell_Eqlib_Potential(HalfCell,F = 96.48534, T_amb = 298.15, R = 0.0083145):
    """
    Help for this function.
    """
    # F = 96.48534 #Faraday's number [kC/equivalence]
    # R = 0.0083145 #Universal gas constant [kJ/mol-K]
    n_elc = HalfCell.n
    #T_amb = 273.15 + 25 #[K]
    T = HalfCell.Temp
     
    Delta_G_cell = np.dot(HalfCell.G,HalfCell.nu)
    Delta_S = np.dot(HalfCell.S,HalfCell.nu)

    U_0_Cell_amb =  -Delta_G_cell/(n_elc*F)
    U_0_Cell = U_0_Cell_amb + (T- T_amb)*Delta_S/(n_elc*F)
    U_Cell = U_0_Cell - R*T/n_elc/F*np.log(np.prod(np.power(HalfCell.X,HalfCell.nu)))
    return U_Cell

def current_density(i_o,V,U,T,F = 96.48534, Beta = 0.5, R = 0.0083145, n = 1):
    """
    The number this returns is absolutly massive, but I cannot find an error in my equation
    
    Parameters
    ----------
    i_o : Exchange Current Density [mA/cm^2] 
    V : Electrode potential difference at the electrode-electrolyte interface [V]
    U : Equilibrium potential [V]
    T : Temperature of the interface [K]    
    F : Optional, Faraday's number
        The default is 96.48534.96.48534 [kC/equivalence]
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
    i= i_o*(math.exp((Beta)*F*n*(V-U)/(R*T)) - math.exp(-(1-Beta)*F*n*(V-U)/(R*T)))
    return i

class Species:
    """
    Defines the thermodynamic properties of the species
    """
    def __init__(self, Name, Gibbs_energy_formation, Standard_Entropy,Standard_State):
        self.name = Name
        self.DG_f = Gibbs_energy_formation
        self.S = Standard_Entropy
        self.state = Standard_State

class Participant(Species):
    """
    Subclass of Species
    Takes in the Species, ionic charge, stoichiometric coefficient, and concentration   
    """
    def __init__(self, Species,charge, stoichiometric_coefficient, concentration):
        super().__init__(Species.name, Species.DG_f, Species.S, Species.state)
        self.charge = charge
        self.stioch_coeff = stoichiometric_coefficient
        self.X = concentration

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
        self.X = [None]*(len(Reactants)+len(Products))
        self.nu = [None]*(len(Reactants)+len(Products))
        for i in Reactants:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.X[indx] = i.X
            self.nu[indx] = i.stioch_coeff*-1
            indx = indx + 1
        for i in Products:
            self.name[indx] = i.name
            self.G[indx] = i.DG_f
            self.S[indx] = i.S
            self.X[indx] = i.X
            self.nu[indx] = i.stioch_coeff
            indx = indx + 1

