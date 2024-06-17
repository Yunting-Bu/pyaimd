import numpy as np
import constants as const

def cal_kin(natm,vel,atom_mass):
    Ekin = 0.0
    for i in range(natm):
        Ekin = Ekin + 0.5*atom_mass[i]*const.dalton2au*np.linalg.norm(vel[i,:])**2.0
    return Ekin

def cal_temp(natm,Ekin):
    temp_cal = (2.0*Ekin)/(3.0*const.kB*const.J2au*natm)
    return temp_cal

def berendsen(dt,bath_temp,con_time,temp_cal):
    f = np.sqrt(1.0+(dt*(bath_temp/temp_cal-1.0)/con_time))
    # From PySCF
    if f > 1.1:
        f = 1.1
    if f < 0.9:
        f = 0.9
    return f 

