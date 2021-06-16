import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy import integrate as ig
from scipy.interpolate import interp1d

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


# Black-body spectrum
def bb(E):
    """
    Function to calculate black-body spectrum
    :param E: list of float energy values [list]
    :return: dataFrame with energy and black-body spectrum values [dataFrame]
    """
    phi_bb_df = pd.DataFrame()
    energy = []
    phi = []
    for e in E:
        phi_bb = ((2*math.pi * e**2)/(h_eV**3 * c**2))/(math.exp(e/(k_eV*T))-1)
        energy.append(e)
        phi.append(phi_bb)
    phi_bb_df['Energy'] = energy
    phi_bb_df['Phi'] = phi
    return phi_bb_df # [1/(eVsm^2)]


# Get AM1.5 spectrum
def getAM15():
    """
    Function to read and return photon flux of AM15 spectrum
    :return: EE: list of energy values (in eV) [list?]
             AM15flux: list of AM1.5 photon flux (in 1/(eV*s*cm^2)) [list?]
    """
    filePath = 'ASTMG173.csv'# 1:wavelength [nm], 3:AM15g spectral power [W/(m^2nm)]
    dataRaw = pd.read_csv(filePath, names=('wavelength','ignore1','AM15pow','ignore2'),skiprows = 2)
    wavelength = dataRaw['wavelength']*1e-9 # nm
    AM15pow = dataRaw['AM15pow']*1e5  # AM1.5 W*cm-2*m-1 (in file: W/(m^2nm))

    EE = h_eV*c/q_eV*1/wavelength # photon energy [eV]
    AM15flux = h_eV*c/q * AM15pow/EE**3 # AM1.5 photon number flux 1/(eV*s*cm^2)
    return EE, AM15flux


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
    Function to calculate the limit of the saturation current density (J0,rad)
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
    J0_integral = result[0]/10 # result[0] = integral result, result[1] = estimate of the absolute error on the result
    return J0_integral


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


# Voltage losses (as defined by Uwe Rau)
def Vloss_Rau(Eg, Voc, Jsc, df=None, voc_rad=None):
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


def Vloss_Vandewal(Jsc, ECT, f, l):
    """
    Function to calculate the radiative Voc loss defined using CT properties
    :param Jsc: measured short-circuit current density [float] [mA/cm**2]
    :param ET: fitted CT state energy [float] [eV]
    :param f: fitted oscillator strength / pre-absorption factor [float] [ev**2]
    :param l: fitted reorganization energy [float] [eV]
    :return: Delta_V_rad: radiative Voc loss [float] [V]
    """

#     Delta_V_rad = k*T/q * math.log((Jsc*h**3*c**2)/(10*f* (q**2) *q*2*math.pi*(ECT-l)* (q))) # multiplied by q to convert eV to J
    V_rad = ECT/q_eV + k_eV*T/q_eV * math.log((Jsc*h_eV**3*c**2)/(10*f*q*2*math.pi*(ECT-l))) # multiplied by q to convert eV to J
    Delta_V_rad_eV = - k_eV*T/q_eV * math.log((Jsc*h_eV**3*c**2)/(10*f*q*2*math.pi*(ECT-l))) # multiplied by q to convert eV to J
    
    return V_rad, Delta_V_rad_eV


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
def calculate_summary(columns, samples, Voc, Jsc, ECT, EIP, f, l):
    """
    Function to calculate summary dataFrame
    :param columns: list of file names [list of strings]
    :param samples: list of EQE files [list of dataFrames]
    :param Voc: list of open-circuit voltage [list of floats]
    :param Jsc: list of short-circuit currents [list of floats]
    :param ECT: list of charge-transfer state values [list of floats]
    :param EIP: list of inflection point values [list of floats]
    :param f: list of oscillator strength values [list of floats]
    :param l: list of reorganization energies [list of floats]
    :return: summary: dataFrame of calculated voltage loss values [dataFrame]
    """
    summary = pd.DataFrame()
    j0_list = []
    led_QE_list = []
    voc_rad_list = []
    delta_voc_rad_list = []
    delta_voc_nonrad_list = []
    
    Voc_SQ_list = []
    Delta_V_SC_Rau_list = []
    Delta_V_rad_Rau_list = []
    Delta_V_nonrad_Rau_list = []
    V_rad_Vandewal_list = []
    Delta_V_rad_Vandewal_list = []
    
    
    for n in range(len(samples)):
        df = samples[n]
        E = df['Energy']
        bb_df = bb(E)

        j0_rad = J0_rad(df, bb_df)
        voc_rad, voc_nonrad = Voc_rad(Voc[n], Jsc[n], j0_rad)
        led_QE = LED_QE(voc_nonrad)

        j0_list.append(j0_rad) #
        voc_rad_list.append(voc_rad) #
        delta_voc_nonrad_list.append(voc_nonrad) #
        led_QE_list.append(led_QE) #
        delta_voc_rad_list.append(ECT[n] - voc_nonrad - Voc[n]) #
        
        Voc_SQ, Delta_V_SC_Rau, Delta_V_rad_Rau, Delta_V_nonrad_Rau = Vloss_Rau(EIP[n], Voc[n], Jsc[n], df=samples[n])
        V_rad_Vandewal, Delta_V_rad_Vandewal = Vloss_Vandewal(Jsc[n], ECT[n], f[n], l[n])

        Voc_SQ_list.append(Voc_SQ) #
        Delta_V_SC_Rau_list.append(Delta_V_SC_Rau) #
        Delta_V_rad_Rau_list.append(Delta_V_rad_Rau) #
        Delta_V_nonrad_Rau_list.append(Delta_V_nonrad_Rau) #

        V_rad_Vandewal_list.append(V_rad_Vandewal) #
        Delta_V_rad_Vandewal_list.append(Delta_V_rad_Vandewal) #

    summary['Sample'] = columns
    summary['Jsc [mA/cm2]'] = Jsc
    summary['ECT [V]'] = ECT
    summary['Voc [V]'] = Voc
    summary['Voc,SQ [V]'] = Voc_SQ_list
    summary['J0,rad [mA/cm2]'] = j0_list
    summary['Voc,rad [V]'] = voc_rad_list
    summary['Voc,rad [V] (Vandewal)'] = V_rad_Vandewal_list
    summary['Delta Voc,SC [V] (Rau)'] = Delta_V_SC_Rau_list
    summary['Delta Voc,rad [V] (Rau)'] = Delta_V_rad_Rau_list
    summary['Delta Voc,rad [V] (Vandewal)'] = Delta_V_rad_Vandewal_list
    summary['Delta Voc,rad [V] (ECT - Delta Voc,rad)'] = delta_voc_rad_list
    summary['Delta Voc,nonrad [V]'] = delta_voc_nonrad_list
    summary['Delta Voc,nonrad [V] (Rau)'] = Delta_V_nonrad_Rau_list
    summary['LED QE'] = led_QE_list

    return summary


# Linear function
def linear(x, m, b):
    """
    Linear function
    :param x: dependent variable [float]
    :param m: slope [float]
    :param b: y-intercept [float]
    :return: independent variable [float
    """
    return m*x + b

# Standard matplotlib plot
def set_up_plot(x_label, y_label, figsize=(8,6), values=None, labels=None):
    """
    Function to set up standard plot
    :param x_label: label for x-axis [str]
    :param y_label: label for y-axis [str]
    :param figsize: figure size [tuple]
    :param values: values for x-axis ticks [list of floats]
    :param labels: labels for x-axis ticks [list of str]
    :return: fig: figure object
    """

    fig = plt.figure(figsize=figsize, dpi=100)

    plt.grid(False)
    plt.tick_params(labelsize=15)
    plt.minorticks_on()
    plt.rcParams['figure.facecolor'] = 'xkcd:white'
    plt.rcParams['figure.edgecolor'] = 'xkcd:white'
    plt.tick_params(labelsize=12, direction='in', axis='both', which='major', length=8, width=2)
    plt.tick_params(labelsize=12, direction='in', axis='both', which='minor', length=0, width=2)

    plt.xlabel(x_label, fontsize=15, fontweight='medium')
    plt.ylabel(y_label, fontsize=15, fontweight='medium')

    if values is not None and labels is not None:
        plt.xticks(values, labels)

    # plt.legend(fontsize=12)  # , loc=2, ncol=2, mode="expand", borderaxespad=0.) # bbox_to_anchor=(0.05, 1.1, 0.9, .102),

    return fig




