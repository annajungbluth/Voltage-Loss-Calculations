import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict
from scipy.optimize import curve_fit
from utils import linear

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


def linear_fit_to_dict(EQE_df, start, stop):
    """
    Function to perform linear fit in dictionary
    :param EQE_df : dataFrame of EQE data [dataFrame]
    :param start : start index of region to fit [int]
    :param stop : stop index of region to fit [int]
    :return data_dict : dictionary of fit information [dict]
    """
    
    data_dict = {}
    
    energy_ = EQE_df['Energy'][start:stop]
    EQE_ =  EQE_df['EQE'][start:stop]
    ppot, pcov = curve_fit(linear, energy_, np.log10(EQE_)) # Fit selected region
    
    data_dict['start_index'] = start
    data_dict['stop_index'] = stop
    data_dict['m'] = ppot[0]
    data_dict['b'] = ppot[1]
    
    return data_dict  


def extend_EQE(orig_EQE, mode, min_energy=0.5, **kwargs):
    """
    Function to extend EQE
    :param orig_EQE : original EQE [dataFrame]
    :param min_energy : minimum energy to extend EQE towards [float]
    :param mode : mode / approach to apply to extend EQE
                  "interpolation" - requires kwargs "left", "right" (constants to insert left & right [float])
                  "linear"        - requires kwargs "start", "stop" (min, max energy for linear fit [float])
                  "inflection"    - can take kwargs "num", "max_infl" (# points for linear fit [int], max energy for inflection ROI [float])
                  "gaussian"      - requires kwargs "CT_df", "n" (dataFrame with CT fit, and sample number [int])
    :return : EQE_new : interpolated EQE [dataFrame]
    """

    if mode=='interpolation': # inerpolate & extend EQE with constants
        
        left = kwargs['left']
        right = kwargs['right']

        interp_range=np.linspace(min_energy, max(orig_EQE['Energy'].values), num=840)
        EQE_interp = np.interp(
            x=interp_range, 
            xp=orig_EQE['Energy'][::-1], 
            fp=orig_EQE['EQE'][::-1], 
            left=left, 
            right=right) # Interpolate x-values and extend left/right with constants
        
        EQE_new = pd.DataFrame()
        EQE_new['Energy'] = interp_range
        EQE_new['EQE'] = EQE_interp

    elif mode=='linear': # Extend low energy regime with linear fit
        
        start = kwargs['start']
        stop= kwargs['stop']
        
        # Pick range for linear fit
        energy_ = orig_EQE['Energy'][orig_EQE['Energy'].between(start, stop)].values
        EQE_ =  orig_EQE['EQE'][orig_EQE['Energy'].between(start, stop)].values

        ppot, pcov = curve_fit(linear, energy_, np.log10(EQE_)) # Fit selected region

        interp_range = np.arange(min_energy, start, 0.01) # Define range to extend the EQE over
        EQE_interp = linear(interp_range, ppot[0], ppot[1])

        EQE_new = pd.DataFrame()
        EQE_new['Energy'] = interp_range[::-1]
        EQE_new['EQE'] = 10**(EQE_interp[::-1])  
        
        stop_EQE = int((orig_EQE['Energy']>start).sum()) # Extract the index where orig_EQE['Energy'] > start
        EQE_new = orig_EQE[:stop_EQE].append(EQE_new) # removing overlap between linear region & EQE
        
    elif mode=='inflection': # Find highest slope between end & inflection point
        
        if 'num' in kwargs.keys():
            num = kwargs['num']
        else:
            num = 5
            
        if 'max_infl' in kwargs.keys():
            max_infl = kwargs['max_infl']
        else:
            max_energy = 1.9
        
        infl = calculate_infl_point(EQE_df = orig_EQE, max_energy = max_infl) 
        infl_index = infl['Index']
        max_index = len(orig_EQE)
        
        fit_dict = defaultdict(list)
        
        for n in range(max_index - infl_index - num):
            start = infl_index+n
            stop = infl_index+num+n
            
            data_dict = linear_fit_to_dict(orig_EQE, start, stop)
            for key, val in data_dict.items():
                fit_dict[key].append(val)

        df_fit = pd.DataFrame(fit_dict)
        best_fit_index = df_fit['m'].idxmax()
        
        interp_range = np.arange(min_energy, orig_EQE['Energy'][df_fit['stop_index'][best_fit_index]], 0.05)
        EQE_interp = linear(interp_range, df_fit['m'][best_fit_index], df_fit['b'][best_fit_index])
       
        EQE_new = pd.DataFrame()
        EQE_new['Energy'] = interp_range[::-1]
        EQE_new['EQE'] = 10**(EQE_interp[::-1])    
        EQE_new = orig_EQE[:int(df_fit['stop_index'][best_fit_index])].append(EQE_new) 

    elif mode=='gaussian': # Extend the EQE with gaussian CT fit
        
        CT_df = kwargs['CT_df']
        n = kwargs['n']
        
        if 'num' in kwargs.keys():
            num = kwargs['num']
        else:
            num = -5
        
        EQE_new = wrapper_extend_EQE_CT(CT_df = CT_df, orig_EQE=orig_EQE, n=n, num=num, min_energy = min_energy)
        
    return EQE_new


# Wrapper function to extend EQE
def wrapper_extend_EQE_CT(CT_df, orig_EQE, n, num=-5, min_energy = 0.5):
    """
    Function to feed parameters into 'extend_EQE' function.
    :param CT_df: DataFrame with information on CT / Opt fit values [dataFrame]
    :param orig_EQE: Original EQE data [dataFrame]
    :param samples: List of EQE files [list of dataFrames]
    :param n: Samples to select [int]
    :param num: Number of data points to discard in original EQE spectrum to connect to CT fits [int]
    :param min_energy: Lowest energy value to extend the EQE to [float]
    :return EQE_extended: dataFrame of extended EQE spectrum [dataFrame]
    """
    
    energy = np.arange(0.5, 2, 0.01)
    CT_EQE_df = Marcus_Gaussian(energy, CT_df['ECT (eV)'][n], CT_df['l_CT (eV)'][n], CT_df['f_CT (eV2)'][n], CT_df['Eopt (eV)'][n], CT_df['l_opt (eV)'][n], CT_df['f_opt (eV2)'][n], 300)
    EQE_extended = extend_EQE_CT(min_energy, orig_EQE, CT_EQE_df, num)
    
    return EQE_extended 


# Function to extend EQE
def extend_EQE_CT(min_energy, orig_EQE_df, CT_df, num=-5):
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
    
#     plt.semilogy(EQE_extended['Energy'], EQE_extended['EQE'])
#     plt.xlim(0.5, 2.5)
    
    return EQE_extended


def calculate_infl_point(EQE_df, start=None, stop=None, max_energy=1.9):
    """
    Function to determine inflection point of the EQE
    :param EQE_df : dataFrame with 'Energy' and 'EQE' values [dataFrame]
    :param start : start index of region of interest [float]
    :param stop : stop index of region of interest [float]
    :param max_energy : the maximum energy to consider in region of interest [float]
    :return infl_dict : dictionary with index and value of inflection point [dict]
    """
    # if no start / stop is given
    if start is None:
        start = min(EQE_df[EQE_df['Energy'] < max_energy].index)
    if stop is None:
        stop = len(EQE_df)-1
  
    Derivative = [(EQE_df['EQE'][x+1]-EQE_df['EQE'][x])/(EQE_df['Energy'][x+1]-EQE_df['Energy'][x]) for x in range(len(EQE_df)-1)]
    Derivative.append(0)
    Derivative_2 = [(Derivative[x+1]-Derivative[x])/(EQE_df['Energy'][x+1]-EQE_df['Energy'][x]) for x in range(len(Derivative)-1)]
    Derivative_2.append(0)
    EQE_df['Derivative'] = Derivative
    EQE_df['Second Derivative'] = Derivative_2
    
    EQE_df.replace([np.inf, -np.inf], np.nan, inplace=True) # replace infinity 

    index = EQE_df['Derivative'][start:stop].idxmax()
    infl = EQE_df['Energy'][index]

    infl_dict = {'Index':index, 'Value':infl}
    
#     plt.plot(EQE_df['Energy'], EQE_df['EQE'])
#     plt.plot(EQE_df['Energy'][index], EQE_df['EQE'][index], marker='*', color='tab:orange')
#     plt.plot(EQE_df['Energy'], EQE_df['Derivative'], color='grey', ls='dotted')
#     plt.plot(EQE_df['Energy'][start], EQE_df['EQE'][start], marker='o', color='tab:blue')
#     plt.plot(EQE_df['Energy'][stop], EQE_df['EQE'][stop], marker='o', color='tab:blue')
    
    return infl_dict