#=======================
# RPA for polymer chains with supramolecular linkages
#  Created by Chris Balzer
#  See top-level README.md for description and citation
#=======================================

# Import packages
import numpy as np

#=======================
# Useful functions
#=======================
# Debye scattering function for continuous Gaussian chain
def debye(x):
    if x == 0:
        return 1.0
    return 2.0/x**2.0 * (np.exp(-x) + x - 1)

# Debye scattering function for discrete Gaussian chain
def debyeDiscrete(x,N):
    Phix = np.exp(-x)
    return (N*(1.0-Phix**2.0) + 2.0*Phix*(Phix**N - 1.0))/(N**2 * (1.0 - Phix)**2)

# Convenient correlation functions
def A(x,si,sj,N):
    return np.exp(-x *(si-sj)/N)

def r(x,s1,s2,N):
    if x == 0:
        return (s1-s2)/N
    return 1.0/x * (1 -  A(x,s1,s2,N))

# Sequence generating functions (continuous chains)
def centerSeq(m,deltaS,N,center=1/2):
    center = N*center
    if m % 2 == 0:
        vec = []
        for i in range(int(m/2)):
            vec.append(center-deltaS/2.0 - i*deltaS)
            vec.append(center+deltaS/2.0 + i*deltaS)
    else:
        vec = [center]
        for i in range(int((m-1)/2)):
            vec.append(center - (i+1)*deltaS)
            vec.append(center + (i+1)*deltaS)
    vec.append(0)
    vec.append(N)
    vec.sort()
    return vec

def endSeq(m,deltaS,N):
    vec = [0]
    for i in range(m):
        vec.append(i*deltaS)
    vec.append(N)
    return vec

# Sequence generating functions (continuous chains)
def sequence(N,M,seq="even",indices=None):
    vec = np.zeros((N+1,1))
    if seq == "even":
        for i in range(0,M):
            vec[int((i+1)*N/(M+1))] = 1
    if seq == "precise":
        if indices is not None:
            for i in indices:
                vec[int(i)] = 1
        else:
            ValueError("Indices must be specified for sequence.")
    if seq == "random":
        vec[np.random.choice(np.arange(1,N,1), M, replace=False)] = 1
    return vec


#=======================
# Structure factor 
#=======================
# Structure factor for continuous chain with supramolecular linkages (Eq. 16 from Balzer and Fredrickson 2024)
def S(k,u0,lam,rhoC,sVec):
    #=======================
    # Input description
    #   u0    --> excluded volume parameter in units of kT*v
    #   lam   --> "bond energy" in units of kT
    #   rhoC  --> reduced chain concentration = n Rg^3/V
    #   sVec  --> Array of functional group positions. Always has form {0, \alpha_1, \alpha_2, ..., \alpha_M, N}. Note the first and last entry are alwats 0 and N.
    #=======================    
    def psiStar0(lam,rhoC,M):
        return (np.sqrt(1.0  + 4.0*lam*M*rhoC) + 1.0)/2.0
    
    # Useful constants
    M   = len(sVec)-2
    N   = sVec[-1]
    x    = k**2.0*N/6.0
    eta  = M*lam*rhoC/psiStar0(lam,rhoC,M)**2
    rho0 = rhoC*N
    
    rSum = 0
    for j in range(1,M+1):
        rSum += r(x,sVec[j],0,N)
        rSum += r(x,N,sVec[j],N)

    denomSum = 0
    for j in range(1,M+1):
        for k1 in range(1,j):
            denomSum += A(x,sVec[j],sVec[k1],N)
        for k1 in range(j+1,M+1):
            denomSum += A(x,sVec[k1],sVec[j],N)

    P = eta/M*rSum**2/(1 - eta/M*denomSum)
    G = debye(x) + P
    
    return rho0*N*G/(1 + u0*rho0*N*G)

# Structure factor for discrete Gaussian chains with supramolecular linkages (Eq. B1 from Balzer and Fredrickson 2024)
def Sd(k,u0,rhoC,lam,alphaVec):
    #=======================
    # Input description
    #   u0    --> excluded volume parameter in units of kT*v
    #   lam   --> "bond energy" in units of kT
    #   rhoC  --> reduced chain concentration = n Rg^3/V
    #   alphaVec  --> Boolean array of functionalized beads. Total length is N.
    #=======================    
    def psiStar0(lam,rhoC,M):
        return (np.sqrt(1.0  + 4.0*lam*M*rhoC) + 1.0)/2.0
    
    # Useful constants
    x = k**2/6.0
    Phix = np.exp(-x)
    
    M = int(np.sum(alphaVec))
    N = len(alphaVec)
    alphaIndx,_ = np.where(alphaVec == 1)
    eta  = M*lam*rhoC/psiStar0(lam,rhoC,M)**2
    rho0 = rhoC*N

    sum1 = 0
    for j in range(M):
        for m in range(1,N+1):
            sum1 +=  Phix**(np.abs(alphaIndx[j] - m))
    sum1 /= N
     
    sum2 = 0.0
    for j in range(M):
        for m in range(M):
            sum2 +=  Phix**(np.abs(alphaIndx[m] - alphaIndx[j]))
        sum2 -= 1.0
    
    P = eta/M*sum1**2/(1 - eta/M*sum2)
    G = debyeDiscrete(x,N) + P
    
    return rho0*N*G/(1.0 + u0*rho0*N*G)

#=======================
# Other structure factor expressions to compare to
#=======================
# Structure factor for continuous Gaussian chain
def Scontinuous(k,u0,rhoC,N):
    rho0 = rhoC * N
    x = k**2 * N/6.0
    G = debye(x)
    return rho0*N*G/(1 + u0*rho0*N*G)

# Structure factor for discrete Gaussian chain
def Sdiscrete(k,u0,rhoC,N):
    rho0 = rhoC * N
    x    =  k**2/6.0
    G    = debyeDiscrete(x,N)
    return rho0*N*G/(1 + u0*rho0*N*G)

# Structure factor for star polymers with N total beads and f arms (from Molina et. al 1997--> https://doi.org/10.1016/S1089-3156(98)00009-9 )
def Sstar(k,u0,rhoC,N,f):
    rho0 = rhoC * N
    Y = k**2 * N/6.0
    G = 2*(Y - f*(1 - np.exp(-Y/f)) + 0.5*f*(f-1)*(1 - np.exp(-Y/f))**2)/Y**2
    return rho0*N*G/(1 + u0*rho0*N*G)

# Structure factor for continuous chain with both ends functionalized (exactly from Eq. 22 in Balzer and Fredrickson 2024 and equivalent to Eq. 53 from Fredrickson and Delaney 2018 --> https://doi.org/10.1063/1.5027582)
def Stele(k,u0,lam,rhoC,N):
    rho0 = rhoC * N
    x = k**2 * N/6.0
    G = debye(x) + 2*r(x,N,0,N)**2/(psiStar0(lam,rhoC,2)**2/(2*lam*rhoC) - np.exp(-x))
    return rho0*N*G/(1 + u0*rho0*N*G)
