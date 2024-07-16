# spm_functions.py
#
#  This file holds utility functions called by the spm model.

def Half_Cell_Eqlib_Potential(HalfCell,F = 96485.34):
    """
    Help for this function.
    """
    # F = 96485.34 #Faraday's number [C/equivalence]
    n_elc = Half_Cell.n

    Delta_G_reactants = 0
    S_reactants = 0
    for i in HalfCell.Reactants:
        S_reactants = S_reactants + i.S
        Delta_G_reactants = Delta_G_reactants + i.DG_f
    Delta_G_products = 0
    S_products = 0
    for i in HalfCell.Products:
        S_products = S_products + i.DG_f
        Delta_G_products = Delta_G_products + i.DG_f
    
    Delta_G_cell = Delta_G_products - Delta_G_reactants
    Delta_S_cell = S_products - S_reactants
    U_0_Cell_25 = Delta_G_cell/(n_elc*F)
    U_0_Cell_T = U_0_Cell_25
    U_Cell = U_0_Cell_T
    return U_Cell


def Interface_Current_Density(HalfCell,V,F = 96485.34,Beta = 0.5,R = 8.314):
    """
    F [C/eqivalence], Beta [-], R [J/K⋅mol]
    """
    F = 96485.34 #Faraday's number [C/equivalence]

class Species:
    """Help for the class"""
    def __init__(self, Name, Gibbs_energy_formation, Standard_Entropy,Standard_State):
        self.name = Name
        self.DG_f = Gibbs_energy_formation
        self.S = Standard_Entropy
        self.state = Standard_State

class Participant(Species):
    """Help for the class"""
    def __init__(self, Species,charge, stoichiometric_coefficient, concentration):
        super().__init__(Species.name, Species.DG_f, Species.S, Species.state)
        self.charge = charge
        self.stioch_coeff = stoichiometric_coefficient
        self.con = concentration

class Half_Cell:
    """Takes in the reactants, products, number of electrons, and the temerature of the half cell in Kelvin"""
    def __init__(self, Reactants,Products,n,Temperature):
        self.Reactants = Reactants
        self.Products = Products
        self.n = n
        self.Temp = Temperature

C6 = Species("C",0,0,1)    
LiC6 = Species("LiC",-230.0,-11.2,1)
Li_plus = Species("Li+",-293.3,49.7,1)
#Li_s = Species("Li(s)",0,0,1)

C6_rxn = Participant(C6,0,1,0.5)
LiC6_rxn = Participant(LiC6,0,1,0.5)
Li_plus_rxn = Participant(Li_plus,1,1,1)
#Li_s_rxn = Participant(Li_s,0,1,1)

React = [Li_plus_rxn,C6_rxn]
Prod = [LiC6_rxn]

#React2 = [LiC6_rxn]
#Prod2 = [Li_s_rxn,C6]

HC = Half_Cell(React,Prod,1,298.15)

U = Half_Cell_Eqlib_Potential(HC)

i_o = 12.3 #exchange current density [mA/cm2]
V = 