from pyscf import gto  
from pyscf import scf  
from pyscf import grad 
import numpy as np
import md_func as aimd
import constants as const

# Read molecular structure from "H2O.xyz"
mol = gto.Mole()
mol.atom = "H2O.xyz"
mol.basis = "sto-3g"
mol.build()

# Calculate energy as potential
mf = scf.RHF(mol)
Epot = mf.kernel()
# Calculate gradient
g = mf.Gradients()
grad = g.kernel()

atom_mass = np.array(mol.atom_mass_list(), dtype=float)
mass_tot = np.sum(atom_mass)   # Store mass in units of dalton

natm = np.size(atom_mass)
# Store elemental symbol
elem = [None]*natm  
for i in range(natm):
    elem[i] = mol.atom_symbol(i)

# Initialize velocity
# Generate random numbers with mean 0, variance 1, following a normal distribution
init_temp = 298.15
vel_old = np.random.normal(0, 1.0, size=np.prod((natm,3))).reshape(natm,3)

# Subracting the average velocity so the center of mass does not move
vel_avg = np.zeros(3, dtype=float)
for i in range(3):
    vel_avg[i] = np.sum(vel_old[:,i])/natm
for i in range(natm):
    vel_old[i,:] = vel_old[i,:] - vel_avg[i]

# Maxwell-Boltzmann
for i in range(natm):
    vel_old[i,:] = np.sqrt((init_temp*const.kB*const.J2au) \
                         /(atom_mass[i]*const.dalton2au))*vel_old[i,:]

pos_old = np.array(mol.atom_coords(), dtype=float)  # Store initial coordinates in units of Bohr

acc_old = np.zeros((natm,3), dtype=float)
pos_new = np.zeros((natm,3), dtype=float)
vel_new = np.zeros((natm,3), dtype=float)
acc_new = np.zeros((natm,3), dtype=float)

# Write coordinates in pos.xyz
with open('pos.xyz', 'w') as f:
    f.write(f'{natm}\n')
    f.write('nstep = 0\n')
    for symbol, coords in zip(elem, pos_old):
        f.write(f'{symbol}  {coords[0]*const.ang2bo:> .6f}  {coords[1]*const.ang2bo:> .6f}  {coords[2]*const.ang2bo:> .6f}\n')

# Time step(fs) and number of steps
dt = 0.5 * const.fs2au
nstep = 1000

# Temperature of the thermostat (K), time constant (fs)
bath_temp = 298.15
con_time = 30.0 * const.fs2au 

# Calculate temperature and kinetic
Ekin = aimd.cal_kin(natm,vel_old,atom_mass)
temp_cal = aimd.cal_temp(natm,Ekin)

# Write AIMD data in aimd.dat
with open('aimd.dat','w') as f:
    f.write(f' step      Ekin           Epot            Etot        temp\n')
    f.write(f'=====  ============  ==============  ==============  ======\n')
    f.write(f'{0:5d}  {Ekin:.10f}  {Epot:.10f}  {Epot+Ekin:.10f}  {temp_cal:.2f}\n')

for n in range(nstep):
    # Update coordinates with Verlet velocity
    for i in range(natm):
        acc_old[i] = - grad[i]/(atom_mass[i]*const.dalton2au)
        pos_new[i,:] = pos_old[i,:] + vel_old[i,:] * dt \
                                    + 0.5 * acc_old[i,:] * (dt**2.0)
    
    # Update coordinates in PySCF
    mol = gto.Mole()
    mol.atom = ''
    for sym, coords in zip(elem, pos_new):
        x, y, z = coords[:3]
        mol.atom += f'{sym} {x} {y} {z}; '  # Note the format: 'symbol x y z;'
    mol.unit = 'B'
    mol.build()
    # Calculate energy and gradients
    mf = scf.RHF(mol)
    Epot = mf.kernel()
    g = mf.Gradients()
    grad = g.kernel()

    # Update velocity with Verlet velocity
    for i in range(natm):
        acc_new[i] = - grad[i]/(atom_mass[i]*const.dalton2au)
        vel_new[i,:] = vel_old[i,:] \
                       + 0.5 * (acc_new[i,:]+acc_old[i,:]) * dt
        
    # Remove translational motion
    mv = np.zeros((3), dtype=float)
    vel_cent = np.zeros((3), dtype=float)

    for i in range(natm):
        mv[:] = mv[:] + vel_new[i,:]*(atom_mass[i]*const.dalton2au)

    vel_cent[:] = mv[:]/(mass_tot*const.dalton2au)
    for i in range(natm):
        vel_new[i,:] = vel_new[i,:] - vel_cent[:]

    # Thermostat:
    Ekin = aimd.cal_kin(natm,vel_old,atom_mass)
    temp_cal = aimd.cal_temp(natm,Ekin)
    f = aimd.berendsen(dt,bath_temp,con_time,temp_cal)
    vel_new = vel_new * f

    # Update pos.xyz and aimd.dat
    with open('pos.xyz', 'a') as f:
        f.write(f'{natm}\n')
        f.write(f'nstep = {n+1}\n')
        for symbol, coords in zip(elem, pos_new):
            f.write(f'{symbol}  {coords[0]*const.ang2bo:> .6f}  {coords[1]*const.ang2bo:> .6f}  {coords[2]*const.ang2bo:> .6f}\n')

    with open('aimd.dat','a') as f:
        f.write(f'{n+1:5d}  {Ekin:.10f}  {Epot:.10f}  {Epot+Ekin:.10f}  {temp_cal:.2f}\n')

    pos_old = pos_new
    vel_old = vel_new

