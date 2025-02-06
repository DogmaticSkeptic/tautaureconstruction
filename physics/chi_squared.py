import numpy as np
from scipy.optimize import minimize

# Constants
m_tau = 1.776  # Tau mass in GeV
m_Z = 91.1876  # Z-boson mass in GeV
sigma_tau = 0.001  # Tau mass uncertainty (GeV)
sigma_MET = 0.01  # MET uncertainty (GeV)
sigma_Z = 0.002  # Z mass uncertainty (GeV)

def chi_squared_nu(neutrino_params, p_pi_p, p_pi_m, MET_x, MET_y):
    """Chi-squared function for neutrino momentum reconstruction"""
    
    # Extract neutrino momenta from optimization parameters
    p_nu_p = np.array([np.linalg.norm(neutrino_params[:3]), *neutrino_params[:3]])  # [E, px, py, pz]
    p_nu_m = np.array([np.linalg.norm(neutrino_params[3:]), *neutrino_params[3:]])  # [E, px, py, pz]

    # Reconstruct tau momenta
    p_tau_p = p_pi_p + p_nu_p
    p_tau_m = p_pi_m + p_nu_m

    # Compute chi-squared terms
    chi2_tau_p = ((m_tau**2 - (p_tau_p[0]**2 - np.sum(p_tau_p[1:]**2)))**2) / sigma_tau**2
    chi2_tau_m = ((m_tau**2 - (p_tau_m[0]**2 - np.sum(p_tau_m[1:]**2)))**2) / sigma_tau**2
    chi2_MET_x = ((p_nu_p[1] + p_nu_m[1] - MET_x)**2) / sigma_MET**2
    chi2_MET_y = ((p_nu_p[2] + p_nu_m[2] - MET_y)**2) / sigma_MET**2
    chi2_Z_mass = ((m_Z**2 - ((p_tau_p + p_tau_m)[0]**2 - np.sum((p_tau_p + p_tau_m)[1:]**2)))**2) / sigma_Z**2

    # Total chi-squared
    chi2_total = chi2_tau_p + chi2_tau_m + chi2_MET_x + chi2_MET_y + chi2_Z_mass

    return chi2_total
