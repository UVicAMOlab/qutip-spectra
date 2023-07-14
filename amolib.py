'''
This is my working library of useful functions and constants of nature

Andrew MacRae. Last Updated July 14, 2023

TODO:
- Animatin helper functions
- Add plotstyle script

'''


# ---  constants ---
amu = 1.66053906660e-27
c = 299792458   # Speed of light
rBohr = 5.2917721090329e-11 # Bohr radius
e = 1.60217662e-19 # elementary charge
hbar = 1.0545718e-34 # hbar, obv
epsilon0 = 8.85418782e-12 
kB = 1.380649e-23; #Boltzmann Constant
Rgas = 8.31446261815324 # ideal gas constant in J/mol*K
Nav = 6.02214076e23 # Avagadro's number
PI = 3.14159265358979 # ... I'm not sure why I did this
me = 9.10938356e-31 # Mass of the electron

light_year = 9460730472580800
astro_unit = 149597870700

# Properties of Rubidium
dRb = 2.8e-29 # dipole moment of Rb *** varies by +/- 0.5e-29 for different transisitions
GammaRb = 2*PI*5.746e6
lambdaRb87_D1 = 794.978851156e-9
lambdaRb87_D2 = 780.241209686e-9

def ndiff(x,dx):
    """ Numerical differentiation of a vector
        Uses symmetric differentiation on middle points, and 3-point method at endpoints
    """
    v = x*0
    sz = x.size
    if(sz < 2):
        return x
    for k in range(2,sz-1):
        v[k] = (x[k+1] - x[k-1])/(2*dx)
    v[0] = (4*x[1]-3*x[0]-x[2])/(2*dx)
    v[sz-1] = -(4*x[sz-2]-3*x[sz-1]-x[sz-3])/(2*dx);
    return v

# AMO functions
def getRb_VapourPressure(T,units = 'Torr',iso = 'Rb85'): # To do: Add option for isotope
    import numpy as np
    """Use Claussius-Clapyron relation to return atomic density for a  given temperature T [Kelvin]"""

    # Allow for eith scalar or vector input
    scl_flg = False 
    if np.isscalar(T):
        scl_flg = True
        T = np.array([T])

    Tk = 273.15 # 0C in K
    Torr2Pa = 133.3223684211 # Conversion of Torr to Pascal
    Tmelt = 39.31 + Tk
    Tboil = 688 + Tk

    # Mask that selects various phases of temperature array
    Tz = T<0
    Ts = (T<=Tmelt)*(T>0)
    Tb = (T<=Tboil)*(T>Tmelt)
    Tl = (T>=Tboil)
    
    Npts = len(T)
    
    logPv = np.ones(Npts)*Tz*1e-9
    logPv += (2.881 + 4.857 - 4215/T)*Ts
    logPv += (2.881 + 4.312 - 4040/T)*Tb
    logPv += (2.881 + 4.312 - 4040/Tboil)*Tl

    if scl_flg:
        P = 10**logPv[0]
    else:
        P = 10**logPv
    if units == 'Pa':
        P *= Torr2Pa
    return P

def getNRb(T):
    P = getRb_VapourPressure(T,units='Pa')
    return (Nav*P)/(Rgas*T)

def chi2lev(freqs,linewidth,rabifreq = 0):
    """ Calculates the susceptibility of a single 2-level atom
        The default value of zero (negligible) Rabi frequency corresponds to purely linear response.
        To incorporate an atomic ensemble, multiply by the atomic density
    """
    chi1 = (dRb**2/(hbar*epsilon0))*1/(1j*GammaRb/2 + freqs)
    chi3 = (2*dRb**4/(epsilon0*hbar**3))*(1j*GammaRb/2 - freqs)/(.25*GammaRb**2 + freqs**2)**2
    return chi1 + chi3*(rabifreq**2)
    
def chi3lev(freqs,Dc,Gs,Gc,Oc,gbc):
    """ Calculates the EIT susceptibility of a 3 level atom in a lambda configuration as seen by a weak probe field
        It is assumed that the probe field being weak means that it is much less than the coupling field
            freqs: frequency vector
            Dc: coupling field detuning
            Gs(c): linewidth of the signal(coupling) field transitions
            Oc: Coupling field Rabi frequency
            gbc: ground-state dephasing
    """
    return (dRb**2/(hbar*epsilon0)) * (freqs - Dc + 1j*gbc)/(Oc**2 - (freqs - Dc + 1j*gbc)*(freqs+1j*(Gs+Gc)/2))

def chi4lev(freqs,Dc,Dm,Gs,Gc,Gm,Oc,Om,gbc):
    """ Calculates the susceptibility of a 4 level atom in N-Type configuration as seen by a weak probe field
        It is assumed that the probe field being weak means that it is much less than the coupling field
            freqs: frequency vector
            Dc: coupling field detuning
            Gs(c)[m]: linewidth of the signal(coupling)[modulation] field transitions
            Oc(m): Coupling(modulation) field Rabi frequency
            gbc: ground-state dephasing
    """
    gEIT = (gbc-1j*(freqs-Dc))
    g3 = (Gc-1j*freqs)
    g4 = (Gm-1j*(Dm-freqs+Dm))
    return 1j*(dRb**2/(hbar*epsilon0))*(gEIT + (Om**2)/g4)/(Oc**2 + g3*gEIT + (Om**2)*g3/g4)