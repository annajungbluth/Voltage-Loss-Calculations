import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# Function to compile gaussian fits
def Marcus_Gaussian(E_, E_CT, l_CT, f_CT, E_opt, l_opt, f_opt, T=300):
    """
    Function to calculate gaussian for CT / Opt fit
    :param E_: list of energy values [list of floats]
    :param ECT: CT state value [float]
    :param l: reorganization energy [float]
    :param f: Oscillation strength [float]
    :return EQE_df: DataFrame of gaussian fit values [dataFrame]
    """
    EQE_df = pd.DataFrame()
    gaussian_CT = [(f_CT/(E*np.sqrt(4*np.pi*l_CT*k_eV*T))*np.exp(-(E_CT+l_CT-E)**2 / (4*l_CT*k_eV*T))) for E in E_]
    gaussian_opt = [(f_opt/(E*np.sqrt(4*np.pi*l_opt*k_eV*T))*np.exp(-(E_opt+l_opt-E)**2 / (4*l_opt*k_eV*T))) for E in E_]
    gaussian_sum = [(f_CT/(E*np.sqrt(4*np.pi*l_CT*k_eV*T))*np.exp(-(E_CT+l_CT-E)**2 / (4*l_CT*k_eV*T)) + f_opt/(E*np.sqrt(4*np.pi*l_opt*k_eV*T))*np.exp(-(E_opt+l_opt-E)**2 / (4*l_opt*k_eV*T))) for E in E_]
    
    EQE_df['Energy'] = E_
    EQE_df['EQE'] = np.array(gaussian_sum)
    EQE_df['EQE (CT)'] = np.array(gaussian_CT)
    EQE_df['EQE (Opt)'] = np.array(gaussian_opt)
    
    return EQE_df


# Wrapper function to extend EQE
def wrapper_extend_EQE(df, samples, n, num=-5, min_energy = 0.5):
    """
    Function to feed parameters into 'extend_EQE' function.
    :param df: DataFrame with information on CT / Opt fit values [dataFrame]
    :param samples: List of EQE files [list of dataFrames]
    :param n: Samples to select [int]
    :param num: Number of data points to discard in original EQE spectrum to connect to CT fits [int]
    :param min_energy: Lowest energy value to extend the EQE to [float]
    :return EQE_extended: dataFrame of extended EQE spectrum [dataFrame]
    """
    
    energy = np.arange(0.5, 2, 0.01)
    CT_EQE_df = Marcus_Gaussian(energy, df['ECT (eV)'][n], df['l_CT (eV)'][n], df['f_CT (eV2)'][n], df['Eopt (eV)'][n], df['l_opt (eV)'][n], df['f_opt (eV2)'][n], 300)
    orig_EQE_df = samples[n]
    EQE_extended = extend_EQE(min_energy, orig_EQE_df, CT_EQE_df, num)
    
    return EQE_extended 


# Function to extend EQE
def extend_EQE(min_energy, orig_EQE_df, CT_df, num=-5):
    """
    Function to extend the EQE by Gaussian shape
    :param min_energy: Lowest energy value to extend the EQE to [float]
    :param orig_EQE_df: Original EQE data to be extended [dataFrame]
    :param CT_df: Data of gaussian CT / Opt fits [dataFrame]
    :param num: Number of data points to discard in original EQE spectrum to connect to CT fits [int]
    :return EQE_extended: dataFrame of extended EQE spectrum [dataFrame]
    """
    
    EQE_func = interp1d(CT_df['Energy'], CT_df['EQE (CT)']) # Create a function to interpolate CT fit values
    
    energy = np.arange(min_energy, min(orig_EQE_df['Energy'][:num]), 0.01)
    energy = energy[::-1] # reverse the order to match EQE order
    
    EQE_add = EQE_func(energy)
    
    d = {'Energy': energy, 'EQE': EQE_add}
    new_df = pd.DataFrame(data = d)

    EQE_cropped_df = orig_EQE_df[:][:num]
    EQE_extended = EQE_cropped_df.append(new_df, ignore_index=True)
    
    plt.semilogy(EQE_extended['Energy'], EQE_extended['EQE'])
    plt.xlim(0.5, 2.5)
    
    return EQE_extended