import math
import pandas as pd
import matplotlib.pyplot as plt

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
