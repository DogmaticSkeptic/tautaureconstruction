import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def compute_four_momentum(vec):
    """Compute four-momentum from particle vector"""
    px, py, pz, m = vec.px, vec.py, vec.pz, vec.mass
    p_magnitude = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_magnitude**2 + m**2)
    return np.array([E, px, py, pz])


# Constants
M_TAU = 1.776  # Tau mass in GeV
M_Z = 91.1876  # Z-boson mass in GeV
SIGMA_TAU = 0.001  # Tau mass uncertainty (GeV)
SIGMA_MET = 0.01  # MET uncertainty (GeV)
SIGMA_Z = 0.002  # Z mass uncertainty (GeV)


# Physics functions
def boost_to_rest_frame(p, p_boost):
    """Boost a 4-momentum p into the rest frame of p_boost"""
    beta = p_boost[1:] / p_boost[0]
    beta_sq = np.dot(beta, beta)

    if beta_sq >= 1.0 or beta_sq < 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])

    gamma = 1.0 / np.sqrt(1.0 - beta_sq)

    beta_dot_p = np.dot(beta, p[1:])
    p_parallel = (beta_dot_p / beta_sq) * beta
    p_perp = p[1:] - p_parallel

    E_prime = gamma * (p[0] - beta_dot_p)
    p_prime_parallel = gamma * (p_parallel - beta * p[0])
    p_prime = np.array([E_prime, *(p_prime_parallel + p_perp)])

    return p_prime

def chi_squared_nu(neutrino_params, p_pi_p, p_pi_m, MET_x, MET_y):
    """Chi-squared function for neutrino momentum reconstruction"""
    p_nu_p = np.array([np.linalg.norm(neutrino_params[:3]), neutrino_params[0], neutrino_params[1], neutrino_params[2]])
    p_nu_m = np.array([np.linalg.norm(neutrino_params[3:]), neutrino_params[3], neutrino_params[4], neutrino_params[5]])

    p_tau_p = p_pi_p + p_nu_p
    p_tau_m = p_pi_m + p_nu_m

    chi2_tau_p = ((M_TAU**2 - (p_tau_p[0]**2 - np.sum(p_tau_p[1:]**2)))**2) / SIGMA_TAU**2
    chi2_tau_m = ((M_TAU**2 - (p_tau_m[0]**2 - np.sum(p_tau_m[1:]**2)))**2) / SIGMA_TAU**2
    chi2_MET_x = ((p_nu_p[1] + p_nu_m[1] - MET_x)**2) / SIGMA_MET**2
    chi2_MET_y = ((p_nu_p[2] + p_nu_m[2] - MET_y)**2) / SIGMA_MET**2
    chi2_Z_mass = ((M_Z**2 - ((p_tau_p + p_tau_m)[0]**2 - np.sum((p_tau_p + p_tau_m)[1:]**2)))**2) / SIGMA_Z**2

    chi2_total = chi2_tau_p + chi2_tau_m + chi2_MET_x + chi2_MET_y + chi2_Z_mass

    return chi2_total


# Compute pseudorapidity function
def compute_eta(p):
    px, py, pz = p[1], p[2], p[3]
    theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
    return -np.log(np.tan(theta / 2))

# Compute phi function
def compute_phi(p):
    px, py = p[1], p[2]
    return np.arctan2(py, px)

# Compute transverse momentum function
def compute_pT(p):
    px, py = p[1], p[2]
    return np.sqrt(px**2 + py**2)

# Analysis functions
def reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y):
    """Reconstruct neutrino momenta using chi-squared minimization"""
    initial_guess = [MET_x / 2, MET_y / 2, 5.0, MET_x / 2, MET_y / 2, -5.0]

    result = opt.minimize(
        chi_squared_nu, initial_guess,
        args=(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y),
        method='BFGS'
    )

    p_nu_p_opt = np.array([np.linalg.norm(result.x[:3]), *result.x[:3]])
    p_nu_m_opt = np.array([np.linalg.norm(result.x[3:]), *result.x[3:]])

    return p_nu_p_opt, p_nu_m_opt


import os

def ensure_plots_dir():
    """Ensure the plots directory exists"""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_comparison_with_ratio(truth_values, reco_values, xlabel, title, bins=50, xlim=None):
    """Plot histograms with ratio plot below and save to file"""
    ensure_plots_dir()
    truth_values = np.array(truth_values)
    reco_values = np.array(reco_values)
    valid_mask = ~np.isnan(truth_values) & ~np.isnan(reco_values)
    truth_values = truth_values[valid_mask]
    reco_values = reco_values[valid_mask]

    if len(truth_values) == 0 or len(reco_values) == 0:
        print(f"Warning: No valid data to plot for {title}")
        return

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    hist_truth, bin_edges = np.histogram(truth_values, bins=bins)
    hist_reco, _ = np.histogram(reco_values, bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax[0].hist(truth_values, bins=bin_edges, alpha=0.5, color='gray', label='Truth')
    ax[0].step(bin_centers, hist_reco, where='mid', color='orange', linewidth=2, label='Reconstructed')

    ax[0].set_ylabel('Count')
    ax[0].set_title(title)
    ax[0].legend()

    ratio = np.divide(hist_reco, hist_truth, out=np.zeros_like(hist_reco, dtype=float), where=hist_truth > 0)

    ax[1].plot(bin_centers, ratio, 'o-', color='black', markersize=4)
    ax[1].axhline(1.0, linestyle='--', color='red', linewidth=1)
    ax[1].set_ylim(0, 2)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Reco / Truth')

    if xlim is not None:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    plt.tight_layout()
    # Create a safe filename from the title
    filename = title.lower().replace(' ', '_').replace('$', '').replace('/', '_') + '.png'
    plt.savefig(f'plots/{filename}')
    plt.close()

def chi_squared_collinear(params, p_pi_p, p_pi_m, MET_x, MET_y):
    """Chi-squared function for collinear neutrino momentum reconstruction"""
    alpha, beta = params
    
    # Reconstructed neutrino momenta (collinear with pions)
    p_nu_p = np.array([alpha*np.linalg.norm(p_pi_p[1:]), alpha*p_pi_p[1], alpha*p_pi_p[2], alpha*p_pi_p[3]])
    p_nu_m = np.array([beta*np.linalg.norm(p_pi_m[1:]), beta*p_pi_m[1], beta*p_pi_m[2], beta*p_pi_m[3]])

    # Tau+ calculations
    p_tau_p = p_pi_p + p_nu_p
    chi2_tau_p = ((M_TAU**2 - (p_tau_p[0]**2 - np.sum(p_tau_p[1:]**2)))**2) / SIGMA_TAU**2
    
    # Tau- calculations
    p_tau_m = p_pi_m + p_nu_m
    chi2_tau_m = ((M_TAU**2 - (p_tau_m[0]**2 - np.sum(p_tau_m[1:]**2)))**2) / SIGMA_TAU**2

    # MET constraints
    chi2_MET_x = ((p_nu_p[1] + p_nu_m[1] - MET_x)**2) / SIGMA_MET**2
    chi2_MET_y = ((p_nu_p[2] + p_nu_m[2] - MET_y)**2) / SIGMA_MET**2

    # Combined Z system
    p_Z = p_tau_p + p_tau_m
    chi2_Z_mass = ((M_Z**2 - (p_Z[0]**2 - np.sum(p_Z[1:]**2)))**2) / SIGMA_Z**2

    # Print individual terms (commented out)
    # print(f"\nChi2 terms for alpha={alpha:.3f}, beta={beta:.3f}:")
    # print(f"Tau+: {chi2_tau_p:.2f}")
    # print(f"Tau-: {chi2_tau_m:.2f}") 
    # print(f"Z: {chi2_Z_mass:.2f}")
    # print(f"MET: {chi2_MET_x + chi2_MET_y:.2f}")

    return chi2_tau_p + chi2_tau_m + chi2_MET_x + chi2_MET_y + chi2_Z_mass

def reconstruct_neutrino_collinear(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y):
    """Collinear approximation reconstruction"""
    result = opt.minimize(chi_squared_collinear, [0, 0], 
                        args=(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y),
                        method='COBYLA', tol=1000,
                        options={'disp': True, 'maxiter': 100})
    
    alpha, beta = result.x
    p_nu_p = np.array([alpha*np.linalg.norm(p_pi_p_reco[1:]), 
                      alpha*p_pi_p_reco[1], 
                      alpha*p_pi_p_reco[2], 
                      alpha*p_pi_p_reco[3]])
    p_nu_m = np.array([beta*np.linalg.norm(p_pi_m_reco[1:]),
                      beta*p_pi_m_reco[1],
                      beta*p_pi_m_reco[2],
                      beta*p_pi_m_reco[3]])
    return p_nu_p, p_nu_m

def plot_residual_comparison(residuals1, residuals2, xlabel, title, truth_values, bins=50, xlim=(-10, 10)):
    """Plot residual relative uncertainty histograms comparing two methods"""
    ensure_plots_dir()
    
    # Calculate relative uncertainties
    rel_residuals1 = [r/truth if truth != 0 else 0 for r, truth in zip(residuals1, truth_values)]
    rel_residuals2 = [r/truth if truth != 0 else 0 for r, truth in zip(residuals2, truth_values)]
    
    plt.figure(figsize=(10,6))
    plt.hist(rel_residuals1, bins=bins, range=xlim, 
            alpha=0.5, label='Original Method')
    plt.hist(rel_residuals2, bins=bins, range=xlim,
            alpha=0.5, label='Collinear Method')
    plt.xlabel(f'Relative {xlabel}')
    plt.ylabel('Count')
    plt.title(f'Relative {title}')
    plt.legend()
    filename = 'relative_' + title.lower().replace(' ', '_').replace('$', '') + '.png'
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_collinearity_test(truth_pion_momenta, truth_neutrino_momenta, particle_type, charge):
    """Plot the collinearity test between truth pions and neutrinos"""
    ensure_plots_dir()
    
    # Calculate dot products of normalized 3-momenta
    collinearity = []
    for pion, nu in zip(truth_pion_momenta, truth_neutrino_momenta):
        pion_3vec = pion[1:] / np.linalg.norm(pion[1:])
        nu_3vec = nu[1:] / np.linalg.norm(nu[1:])
        collinearity.append(np.dot(pion_3vec, nu_3vec))
    
    plt.figure(figsize=(10,6))
    plt.hist(collinearity, bins=50, range=(-1,1), alpha=0.7)
    plt.xlabel('Collinearity (cosθ)')
    plt.ylabel('Count')
    plt.title(f'Collinearity Test for {particle_type}{charge}')
    plt.grid(True)
    filename = f'collinearity_test_{particle_type}_{charge}.png'
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_relative_uncertainty(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=(-3, 3)):
    """Plot relative uncertainties between truth and reconstructed values and save to file"""
    ensure_plots_dir()
    rel_unc = [(reco - truth) / truth if truth != 0 else 0
               for truth, reco in zip(truth_values, reco_values)]

    # Create bins that span exactly the xlim range
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)

    plt.figure(figsize=(10, 6))
    plt.hist(rel_unc, bins=bin_edges, alpha=0.7)
    plt.xlabel(f'Relative Uncertainty in {component}')
    plt.ylabel('Count')
    plt.title(rf'Relative Uncertainty in {component} for {particle_type}{charge}')
    plt.xlim(xlim)
    plt.grid(True)
    filename = f'rel_unc_{particle_type}_{charge}_{component}.png'
    plt.savefig(f'plots/{filename}')
    plt.close()
