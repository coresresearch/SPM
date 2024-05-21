# spm.py
#
# This file serves as the main model file.  
#   It is called by the user to run the mode
# 
from spm_functions import Half_Cell_Eqlib_Potential,Butler_Volmer,Species,Participant,Half_Cell

C6 = Species("C",0,0,1,0)    
LiC6 = Species("LiC",-230.0,-11.2,1,0)
Li_plus = Species("Li+",-293.3,49.7,1,1)

C6_rxn = Participant(C6,1,0.5)
LiC6_rxn = Participant(LiC6,1,0.5)
Li_plus_rxn = Participant(Li_plus,1,1)

React = [Li_plus_rxn,C6_rxn]
Prod = [LiC6_rxn]

HC = Half_Cell(React,Prod,1,298.15)

U = Half_Cell_Eqlib_Potential(HC)
print(U,"[V]")

I = current_density(12.3,-0.8,U,HC.Temp)
print(I,"[mA/cm2]")