#=======================
# RPA for polymer chains with supramolecular linkages
#  Created by Chris Balzer
#  See top-level README.md for description and citation
#=======================================

import matplotlib.pyplot as plt
from polymerSolution import *

# Make plots pretty
plt.style.use('seaborn-v0_8-dark-palette')
plt.rcParams.update({
        "font.weight": "bold",  # bold fonts
        "lines.linewidth": 2,   # thick lines
        "lines.color": "k",     # black lines
        "savefig.dpi": 300,     # higher resolution output
        'text.usetex': True,
        "font.family": "serif",
        "font.serif": ['Computer Modern'],
        "font.size": 10,
        "legend.fontsize": 8
    })

#========================
# Set parameters/conditions for all comparisons
#========================
N    = 100
M    = 3
rhoC = 1.0
BC   = 1.0

rho0 = rhoC * N
u0   = BC/(rho0*N) 
Rg0  = np.sqrt(N/6)

# Set sequences
contSeq     = centerSeq(M,N/20,N,center=1/2)
discreteSeq = sequence(N,None,"precise",indices=[int(x) for x in contSeq[1:-1]])

# Range of k values
kVec  = np.arange(0.001,5.0/Rg0,0.01)

#========================
# Compare discrete and continuous structure factors
#========================
# Set "bond strength"
lam = 1

# Initialize figure
f     = plt.figure(figsize=(5, 4))

# Continuous Gaussian Chain
Svals = 0*kVec
for k in range(len(kVec)):
    Svals[k] = S(kVec[k],u0,lam,rhoC,contSeq)
plt.plot(kVec*Rg0,Svals/(rho0*N),'-',color='black',label=r"CGC")

# Discrete Gaussian Chain
Svals = 0*kVec
for k in range(len(kVec)):
    Svals[k]  = Sd(kVec[k],u0,rhoC,lam,discreteSeq)
plt.plot(kVec*Rg0,Svals/(rho0*N),'--',color='red',label=r"DGC")


plt.ylim([0, 1.2])
plt.xlim([0, 5])
plt.ylabel(r'$S(k)/(\rho_0 N)$')
plt.xlabel(r'$k R_g$')
plt.legend(frameon=False)
f.savefig('figures/Compare_Continuous_Discrete.png', bbox_inches='tight')


#========================
# Structure factor for continuous chain with several lambdas
#========================
# List of "bond strength" values
lams = [0, 1, 5, 10, 15, 19.55]

# Initialize figure
f     = plt.figure(figsize=(5, 4))

# Loop through lambda values
for lam in lams:
    Svals = 0*kVec
    for k in range(len(kVec)):
        Svals[k] = S(kVec[k],u0,lam,rhoC,contSeq)
    plt.plot(kVec*Rg0,Svals/(rho0*N),'-',label=r"%.2f"%lam)

plt.ylim([1e-2, 1e5])
plt.xlim([0, 5])
plt.yscale('log')
plt.ylabel(r'$S(k)/(\rho_0 N)$')
plt.xlabel(r'$k R_g$')
plt.legend(frameon=False,title=r"$\lambda$")
f.savefig('figures/Microphase_Example_Continuous.png', bbox_inches='tight')