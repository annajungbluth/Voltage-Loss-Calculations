import math
import pandas as pd
import numpy as np
from scipy import integrate as ig
from scipy.interpolate import interp1d
from utils import bb, getAM15, linear

# Import Important Parameters

T = 293 # [K] ambient & cell temperature
h = 6.626 * 10**(-34) # [kgm^2/s]
h_eV = 4.1357*10**(-15) # eV s
c = 2.998 * 10**(8) # [m/s]
k = 1.3806 * 10**(-23) # [kgm^2/s^2K]
k_eV = 8.6173*10**(-5) # eV / K
q = 1.60217662 * 10**(-19) # [C]
q_eV = 1
Vth = k_eV*T/q_eV # thermal voltage [eV]

# Shockley-Queisser limit
def SQ(Eg):
    """
    Function to calculate Jsc, Voc, FF, and PCE of Shockley-Queisser limit for specific 'energy of reference'
    The 'energy of reference' for original SQ theory is the band gap, which can also be the inflection point of the EQE, or the CT state (?)
    :param Eg: band gap [float]
    :return: Voc: open-circuit voltage [float]
             Jsc: short-circuit current [float]
             FF: fill factor [float]
             PCE: photo conversion efficiency [float]
    """
    energy, AM15flux = getAM15() # lists
    bbcell = bb(energy) # black-body radiation from cell

    Jsc = -1e3*q*ig.simps(AM15flux[energy>=Eg],energy[energy>=Eg]) # mA/cm^2 (- because energy is descending data)
    J0 = -1e-1*q*ig.simps(bbcell['Phi'][energy>=Eg],energy[energy>=Eg]) # mA/cm^2 (bb returns ~1/m^2)
    
    Voc = Vth*math.log(Jsc/J0+1) # +1 negligible
    Vocnorm=Voc/(Vth)
    FF = (Vocnorm-math.log(Vocnorm+0.72))/(Vocnorm+1) # For accuracy of expression see Neher et al. DOI: 10.1038/srep24861
                                                      # (main & supplement S4) very accurate down to Vocn=5: corresponds to Voc=130mV for nid=1 or Voc=260mV for nid=2
    PCE = FF*Voc*Jsc
    FF = FF*100
    
    return Voc,Jsc,FF,PCE


# Calculating the short-circuit current density
def calculate_Jsc(E, EQE): # double check that this is correct!!!
    """
    Function to calculate the limit of the short-circuit current density
    :param E: list of energy values [list of floats]
    :param EQE: list of EQE values [list of floats]
    :return: Jsc: short-circuit current density [float]
    """
    
    energy, AM15flux = getAM15() # lists

    EQE_intp = interp1d(E, EQE)
    AM15_intp = interp1d(energy.values, AM15flux.values)    
    
    result = ig.quad(lambda e: q*EQE_intp(e)*AM15_intp(e), min(E), max(E))
    Jsc = result[0]*(1e3) # to convert into right units?
    
    return Jsc


# Radiative limit of saturation current density
def J0_rad(EQE_df, phi_bb_df):
    """
    Function to calculate the radiative limit of the saturation current density (J0,rad)
    :param EQE_df: dataFrame of EQE spectra, with columns 'Energy' and 'EQE' [dataFrame of floats]
    :param phi_bb_df: dataFrame of black-body spectrum, with columns 'Energy' and 'Phi' [dataFrame of floats]
    :return: J0_rad: radiative limit of the saturation current density [float]
    """
    
    EQE_intp = interp1d(EQE_df['Energy'].values, EQE_df['EQE'].values)
    Phi_intp = interp1d(phi_bb_df['Energy'].values, phi_bb_df['Phi'].values)

#     J0_rad_list = []    
#     for n in range(1,len(EQE_df['Energy'])):
#         j0_rad = q*EQE_df['EQE'][n]*phi_bb_df['Phi'][n]*(EQE_df['Energy'][n-1]-EQE_df['Energy'][n])
#         J0_rad_list.append(j0_rad) # [A / m^2]
#         J0_rad = np.sum(J0_rad_list)/10 # [mA / cm^2]
        
    result = ig.quad(lambda e: q*EQE_intp(e)*Phi_intp(e), min(EQE_df['Energy']), max(EQE_df['Energy']))
    J0_rad_integral = result[0]/10 # result[0] = integral result, result[1] = estimate of the absolute error on the result
    return J0_rad_integral


# Limit of saturation current density
def J0(EQE_df, phi_bb_df, EQE_EL):
    """
    Function to calculate the limit of the saturation current density (J0)
    :param EQE_df: dataFrame of EQE spectra, with columns 'Energy' and 'EQE' [dataFrame of floats]
    :param phi_bb_df: dataFrame of black-body spectrum, with columns 'Energy' and 'Phi' [dataFrame of floats]
    :param EQE_EL: LED quantum efficiency [float]
    :return: J0: limit of the saturation current density [float]
    """
    
    EQE_intp = interp1d(EQE_df['Energy'].values, EQE_df['EQE'].values)
    Phi_intp = interp1d(phi_bb_df['Energy'].values, phi_bb_df['Phi'].values)    

#     J0_list = []
#     for n in range(1,len(EQE_df['Energy'])):
#         j0 = (q/EQE_EL)*EQE_df['EQE'][n]*phi_bb_df['Phi'][n]*(EQE_df['Energy'][n-1]-EQE_df['Energy'][n])
#         J0_list.append(j0) # [A / m^2]
#         J0 = np.sum(J0_list)/10 # [mA / cm^2]

    result = ig.quad(lambda e: (q/EQE_EL)*EQE_intp(e)*Phi_intp(e), min(EQE_df['Energy']), max(EQE_df['Energy']))
    J0_integral = result[0]/10 # result[0] = integral result, result[1] = estimate of the absolute error on the result. Divide by 10 to convert from mA/cm2 to A/m2?
    return J0_integral


# Limit of saturation current density based on CT properties
def J0_CT(EQE_EL, ECT, l, f):
    """
    Function to calculate the limit of the saturation current density (J0) based on CT properties as defined by Koen Vandewal
    :param EQE_EL: LED quantum efficiency (unitless) [float]
    :param ECT: CT state energy(eV) [float]
    :param l: Reorganization energy (eV) [float]
    :param f: Oscillator strength (eV**2) [float]
    :return: J0: limit of the saturation current density [float]
    """
    
    J0 = (q/(10*EQE_EL))*f*2*np.pi/(h_eV**3 * c**2)*(ECT-l)*np.exp(-(ECT/(k_eV*T)))
    return J0


# Function to calculate Voc
def Voc(Jsc, J0):
    """
    Function to calculate Voc from Jsc and J0
    :param Jsc: short-circuit current density [float] [mA/cm2]
    :param J0: saturation current density [float] [A/m2]
    """
    Voc = k_eV*T/q_eV * np.log((10*Jsc/J0)+1)
    return Voc


# Radiative limit of the open-circuit voltage
def Voc_rad(Voc, Jsc, J0_rad):
    """
    Function to calculate the radiative limit of the open-circuit voltage
    :param Voc: measured open-circuit voltage [float]
    :param Jsc: measured short-circuit current [float]
    :param J0_rad: calculated radiative limit of the saturation current density [float]
    :return: Voc_rad: Radiative upper limit of the open-circuit voltage [float]
             Delta_Voc_nonrad: Non-radiative voltage losses [float]
    """
    Voc_rad = k*T/q * math.log((Jsc/J0_rad)+1)
    Delta_Voc_nonrad = Voc_rad - Voc
    return Voc_rad, Delta_Voc_nonrad



# Voltage losses based on SQ theory (as defined by Uwe Rau)
def Vloss_SQ(Eg, Voc, Jsc, df=None, voc_rad=None):
    """
    Function to calculate Shockley-Queisser Voc, voltage losses due to Jsc, radiative, and non-radiative recombination as defined by Rau
    :param Eg: reference energy (could be inflection point of lin. EQE, optical gap, or E_CT) [float]
    :param Voc: open-circuit voltage [float]
    :param Jsc: short-circuit current [float]
    :param df: EQE data with columns 'Energy' and 'EQE' [dataFrame]
    :param Voc_rad: Radiative upper limit of Voc, calculated from sEQE / EL [float]
    :return: Voc_SQ: Voc Shockley-Queisser limit [float]
             DeltaV_sc: Losses due to non-ideal Jsc [float]
             Delta_V_rad: Radiative losses [float]
             Delta_V_nonrad: Non-radiative losses [float]
    """
    if voc_rad is None and df is not None:
        E = df['Energy']
        bb_df = bb(E)
        j0_rad = J0_rad(df, bb_df)
        voc_rad, Delta_V_nonrad = Voc_rad(Voc, Jsc, j0_rad)
        
    Voc_SQ, Jsc_SQ, FF_SQ, PCE_SQ = SQ(Eg)
    Delta_V_sc = k*T/q * math.log(Jsc_SQ/Jsc) # losses due to non-ideal Jsc
    Delta_V_rad = Voc_SQ - voc_rad - Delta_V_sc # radiative losses
    Delta_V_nonrad = voc_rad - Voc # non-radiative losses
    return Voc_SQ, Delta_V_sc, Delta_V_rad, Delta_V_nonrad


# Voltage losses based on CT properties (as defined by Koen Vandewal)
def Vloss_CT(Jsc, Voc, ECT, f, l):
    """
    Function to calculate the radiative Voc loss defined using CT properties
    :param Jsc: measured short-circuit current density [float] [mA/cm**2]
    :param ECT: fitted CT state energy [float] [eV]
    :param f: fitted oscillator strength / pre-absorption factor [float] [ev**2]
    :param l: fitted reorganization energy [float] [eV]
    :return: Delta_V_rad: radiative Voc loss [float] [V]
    """

#     Delta_V_rad = k*T/q * math.log((Jsc*h**3*c**2)/(10*f* (q**2) *q*2*math.pi*(ECT-l)* (q))) # multiplied by q to convert eV to J
    V_rad = ECT/q_eV + k_eV*T/q_eV * math.log((10*Jsc*h_eV**3*c**2)/(f*q*2*math.pi*(ECT-l))) # multiplied by q to convert eV to J
    Delta_V_rad_eV = - k_eV*T/q_eV * math.log((10*Jsc*h_eV**3*c**2)/(f*q*2*math.pi*(ECT-l))) # multiplied by q to convert eV to J
    Delta_V_nonrad = V_rad - Voc
    
    return V_rad, Delta_V_rad_eV, Delta_V_nonrad

# Function to calculate Voc from CT properties
def Voc_CT(EQE_EL, Jsc, ECT, f, l, T=300):
    """
    Function to calculate the Voc defined using CT properties
    :param EQE_EL: LED quantum efficiency [float]
    :param Jsc: measured short-circuit current density [float] [mA/cm**2]
    :param ECT: fitted CT state energy [float] [eV]
    :param f: fitted oscillator strength / pre-absorption factor [float] [ev**2]
    :param l: fitted reorganization energy [float] [eV]
    :return: Voc: Open circuit voltage [float] [V]
    """
    Voc = (ECT/q_eV)+(k_eV*T/q_eV)*np.log((10*Jsc*h_eV**3*c**2)/(f*q*2*np.pi*(ECT-l)))+(k_eV*T)/q_eV * np.log(EQE_EL)
    return Voc


# LED Quantum Efficiency
def LED_QE(Delta_Voc_nonrad):
    """
    Function to calculate LED quantum efficiency
    :param Delta_Voc_nonrad: non-radiative voltage losses [float]
    :return: LED_QE: LED quantum efficiency [float]
    """
    LED_QE = math.exp(-(Delta_Voc_nonrad*q)/(k*T))
    return LED_QE


# Calculate voltage loss summary
# ADJUST to include other loss functions!
def calculate_summary(columns, samples, Voc, Jsc, ECT, Eopt, f, l):
    """
    Function to calculate summary dataFrame
    :param columns: list of file names [list of strings]
    :param samples: list of EQE files [list of dataFrames]
    :param Voc: list of open-circuit voltage [list of floats]
    :param Jsc: list of short-circuit currents [list of floats]
    :param ECT: list of charge-transfer state values [list of floats]
    :param Eopt: list of estimated optical gaps [list of floats]
    :param f: list of oscillator strength values [list of floats]
    :param l: list of reorganization energies [list of floats]
    :return: summary: dataFrame of calculated voltage loss values [dataFrame]
    """
    summary = pd.DataFrame()

    j0_list = []
    j0_rad_list = []
    j0_CT_list = []
    led_QE_list = []

    V_rad_list = []
    Delta_V_rad_list = []
    Delta_V_nonrad_list = []
    
    Voc_SQ_list = []
    Delta_V_SC_SQ_list = []
    Delta_V_rad_SQ_list = []
    Delta_V_nonrad_SQ_list = []

    V_rad_CT_list = []
    Delta_V_rad_CT_list = []
    Delta_V_nonrad_CT_list = []
    
    
    for n in range(len(samples)):
        df = samples[n]
        E = df['Energy']
        bb_df = bb(E)

        # Calculate Parameters        
        j0_rad = J0_rad(df, bb_df)
        
        voc_rad, voc_nonrad = Voc_rad(Voc[n], Jsc[n], j0_rad)
        led_QE = LED_QE(voc_nonrad)
        
        j0 = J0(df, bb_df, led_QE)
        
        Voc_SQ, Delta_V_SC_SQ, Delta_V_rad_SQ, Delta_V_nonrad_SQ = Vloss_SQ(Eopt[n], Voc[n], Jsc[n], df=samples[n])
        V_rad_CT, Delta_V_rad_CT, Delta_V_nonrad_CT = Vloss_CT(Jsc[n], Voc[n], ECT[n], f[n], l[n])
        
        j0_CT = J0_CT(led_QE, ECT[n], l[n], f[n])

        # Add them to lists
        j0_rad_list.append(j0_rad) # Radiative limit of the saturation current density
        j0_list.append(j0) # Limit of the saturation current density
        j0_CT_list.append(j0_CT) # Limit of the saturation current density based on CT properties
        led_QE_list.append(led_QE) # LED Quantum Efficiency

        V_rad_list.append(voc_rad) # Radiative limit of the Voc
        Delta_V_nonrad_list.append(voc_nonrad) # Non-radiative losses
        Delta_V_rad_list.append(ECT[n] - voc_nonrad - Voc[n]) # Radiative losses

        Voc_SQ_list.append(Voc_SQ) # Shockley-Queisser limit of the Voc
        Delta_V_SC_SQ_list.append(Delta_V_SC_SQ) # Losses due to imperfect short-circuit current
        Delta_V_rad_SQ_list.append(Delta_V_rad_SQ) # Radiative losses
        Delta_V_nonrad_SQ_list.append(Delta_V_nonrad_SQ) # Non-radiative losses

        V_rad_CT_list.append(V_rad_CT) # Radiative limit based on CT properties
        Delta_V_rad_CT_list.append(Delta_V_rad_CT) # Radiative losses based on CT properties
        Delta_V_nonrad_CT_list.append(Delta_V_nonrad_CT) # Non-radiative losses based on CT properties

    summary['Sample'] = columns
    summary['Voc [V]'] = Voc
    summary['ECT [V]'] = ECT
    summary['Jsc [mA/cm2]'] = Jsc
    summary['Voc,SQ [V]'] = Voc_SQ_list
    summary['Voc,rad [V]'] = V_rad_list # Comparing Jsc to J0,rad
#     summary['Delta Voc,nonrad [V]'] = Delta_V_nonrad_list # Comparing radiative limit to real Voc
    summary['Delta Voc,SC [V] (Rau)'] = Delta_V_SC_SQ_list # Losses due to imperfect Jsc
    summary['Delta Voc,rad [V] (Rau)'] = Delta_V_rad_SQ_list # Radiative losses
#    summary['Delta Voc,rad [V] (ECT - Delta Voc,rad)'] = Delta_V_rad_list
    summary['Delta Voc,nonrad [V] (Rau)'] = Delta_V_nonrad_SQ_list # Nonradiative losses (should be the same as above)
    summary['Voc,rad [V] (CT properties)'] = V_rad_CT_list  # Radiative limit based on CT properties
    summary['Delta Voc,rad [V] (CT properties)'] = Delta_V_rad_CT_list  # Radiative losses with ECT as upper limit
    summary['Delta Voc,nonrad [V] (CT properties)'] = Delta_V_nonrad_CT_list # Nonradiative losses based on CT properties
    summary['LED QE'] = led_QE_list # LED Quantum Efficiency
    summary['J0 [mA/cm2]'] = j0_list # Saturation current density
    summary['J0,rad [mA/cm2]'] = j0_rad_list # Radiative saturation current density
    summary['J0 (CT properties) [mA/cm2]'] = j0_CT_list # Radiative saturation current density

    return summary






