# Something is wrong here, I will take a deeper look tomorrow, but for now I am curious if my general approach is working

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Held Constant
i_o = 120 # Exchange Current Density [A/m^2] 
C_dl = 6*10**3 # Double Layer Capacitance [F/m^2]
T = 273.15 + 25 # standard temperature [K]
U = 1 # Open circuit (equalibrium) potential [V]
i_ext = 4 # external current [A]
F = 96.48534 #Faraday's number [kC/equivalence]
R = 0.0083145 #Universal gas constant [kJ/mol-K]
n = 1 # number of electrons [-]
Beta = 0.5 # [-] Beta = (1 - Beta) in this case 
Const = Beta*F*n/R/T # combine constants for simplicity

# Geometry
r = 50*10**-6 # Anode radius [m]
V = (4*np.pi/3)*r**3 # Volume of the anode [m^3]
A_surf = 4*np.pi*r**2 # surface area of the anode [m^2]

## Derivation 
# Eta_an = Phi_an - Phi_el - U ; Phi_an = 0 ; Phi_an = Phi_el - delta_Phi_dl_an
# Eta_an = - delta_Phi_dl_an - U => sub into Butler Volmer
# i_ext = i_dl + i_far ; -i_dl = C_dl*(d Delta_Phi_dl/ dt)
# (i_far - i_ext)/C_dl = d Delta_Phi_dl/dt
def set_up_ivp(t,V,Con,C,i_o,A_surf): # returns an expression for d Delta_Phi_dl/dt in terms of Delta_Phi_dl
    i_far= i_o*A_surf*(math.exp(Con*(V-U)) - math.exp(-Con*(V-U)))
    return (i_far - i_ext)/C

# Integration
t_start = 0.5 # [s]
t_end = 1000 # length of time passed in the integration [s]
SV_0 = 0 # intial value for Phi_dl [V]
N = 10000 # number of time steps
dt = t_end/N
t_eval=np.linspace(0,t_end,N)
dt_2 = t_eval[0] - t_eval[1]
print(dt_2)
SV = solve_ivp(lambda t, delta_Phi_dl: set_up_ivp(t,-delta_Phi_dl,Const,C_dl,i_o,A_surf),[t_start,t_end],[SV_0],t_eval=np.linspace(t_start,t_end,N))

# Post Processing
plt1 = plt.figure(1)
plt.plot(SV.t,SV.y[0])
plt.xlabel("time [s]")
plt.ylabel("Change in Potential Across the Double Layer [V]")
plt1.show()

'''
these plots are in progress and not working

plt2 = plt.figure(2)
i_dl = -C_dl*np.gradient(SV.y[0],dt)
i_fa = np.zeros(len(i_dl))
for ind, ele in enumerate(i_fa):
    i_fa[ind] = i_o*A_surf*(math.exp(Const*(-SV.y[0,ind]-U)) - math.exp(-Const*(-SV.y[0,ind]-U)))

plt.plot(np.divide(SV.t,i_dl,i_fa))
plt.plot(SV.t,i_dl)
plt.plot(SV.t,i_ext*np.ones(len(i_dl)))
#plt.plot(SV.t,i_fa)
plt.xlabel("time [s]")
plt.ylabel("Double Layer Current [A]")
#plt.ylim((0,10))
#plt.legend("DL","Ext")
plt2.show()
'''

input()

