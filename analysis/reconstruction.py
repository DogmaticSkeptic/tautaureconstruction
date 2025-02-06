import numpy as np
import scipy.optimize as opt
from physics.chi_squared import chi_squared_nu

def reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y):
    """Reconstruct neutrino momenta using chi-squared minimization"""
    # Initial guess: Set neutrino momenta to half MET for transverse components and 5 GeV in z
    initial_guess = [MET_x/2, MET_y/2, 5.0, MET_x/2, MET_y/2, -5.0]

    # Minimize chi-squared
    result = opt.minimize(chi_squared_nu, initial_guess, 
                         args=(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y), 
                         method='BFGS')

    # Extract optimized neutrino momenta
    p_nu_p_opt = np.array([np.linalg.norm(result.x[:3]), *result.x[:3]])
    p_nu_m_opt = np.array([np.linalg.norm(result.x[3:]), *result.x[3:]])

    return p_nu_p_opt, p_nu_m_opt
