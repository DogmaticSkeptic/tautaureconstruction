import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Constants
m_tau = 1.776  # Tau mass in GeV
m_Z = 91.1876  # Z-boson mass in GeV
sigma_tau = 0.001  # Tau mass uncertainty (GeV)
sigma_MET = 0.01  # MET uncertainty (GeV)
sigma_Z = 0.002  # Z mass uncertainty (GeV)


# Physics functions
def boost_to_rest_frame(p, p_boost, debug=False):
    """Boost a 4-momentum p into the rest frame of p_boost"""
    # Check for invalid inputs
    if np.any(np.isnan(p)) or np.any(np.isnan(p_boost)):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Early-out for trivial boost
    if np.linalg.norm(p_boost[1:]) < 1e-10:
        return p  # Already in the rest frame
    
    # Calculate boost vector (direction and magnitude)
    beta = p_boost[1:] / p_boost[0]  # β = p/E
    beta_sq = np.dot(beta, beta)
    
    # Check for invalid beta_sq
    if beta_sq >= 1.0 or beta_sq < 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    
    # Calculate components parallel and perpendicular to boost
    beta_dot_p = np.dot(beta, p[1:])
    p_parallel = (beta_dot_p / beta_sq) * beta
    p_perp = p[1:] - p_parallel
    
    # Apply Lorentz transformation
    E_prime = gamma * (p[0] - beta_dot_p)
    p_prime_parallel = gamma * (p_parallel - beta * p[0])
    p_prime = np.array([E_prime, *(p_prime_parallel + p_perp)])
    
    return p_prime

def boost_three_vector(vec3, p_boost):
    """Boost a 3-vector by treating it as a 4-vector with E=0"""
    # Check for invalid inputs
    if np.any(np.isnan(vec3)) or np.any(np.isnan(p_boost)):
        return np.array([np.nan, np.nan, np.nan])
    
    # Calculate boost parameters
    beta = p_boost[1:] / p_boost[0]
    beta_sq = np.dot(beta, beta)
    
    if beta_sq >= 1.0:  # also covers beta_sq < 0 implicitly
        return np.array([np.nan, np.nan, np.nan])
    
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    
    # Form the 4-vector with E=0:
    V = np.concatenate(([0.0], vec3))
    
    # Boost the 4-vector: note the standard formulas
    V0_prime = -gamma * np.dot(beta, vec3)
    V_space_prime = vec3 + ((gamma - 1.0) * np.dot(beta, vec3) / beta_sq) * beta
    
    # Now extract and renormalize the spatial part:
    v_prime = V_space_prime
    norm = np.linalg.norm(v_prime)
    if norm > 0:
        return v_prime / norm
    else:
        return np.array([np.nan, np.nan, np.nan])

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

def define_coordinate_system(p_tau_p, p_tau_m):
    """Define the {r^, n^, k^} coordinate system in the tau tau rest frame"""
    # k^ is the flight direction of tau- in the tau tau rest frame
    k_hat = p_tau_m[1:] / np.linalg.norm(p_tau_m[1:])
    
    # p^ is the direction of one of the e± beams (assume z-axis)
    p_hat = np.array([0, 0, 1])
    
    # r^ = (p^ - k^ * cosΘ) / sinΘ
    cos_theta = np.dot(k_hat, p_hat)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Handle case where sinΘ is very small (tau nearly parallel to beam)
    if sin_theta < 1e-6:
        # Use x-axis as reference direction instead of z-axis
        p_hat = np.array([1, 0, 0])
        cos_theta = np.dot(k_hat, p_hat)
        sin_theta = np.sqrt(1 - cos_theta**2)
    
    r_hat = (p_hat - k_hat * cos_theta) / sin_theta
    
    # n^ = k^ × r^
    n_hat = np.cross(k_hat, r_hat)
    
    return r_hat, n_hat, k_hat

# Analysis functions
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

def compute_cos_theta(p_pion, r_hat, n_hat, k_hat):
    """Calculate cos theta for each axis in the rest frame"""
    # Check for invalid inputs
    if np.any(np.isnan(p_pion)) or np.any(np.isnan(r_hat)) or \
       np.any(np.isnan(n_hat)) or np.any(np.isnan(k_hat)):
        return np.nan, np.nan, np.nan
    
    # Calculate pion momentum norm
    p_norm = np.linalg.norm(p_pion[1:])
    if p_norm < 1e-10:  # Avoid division by zero
        return np.nan, np.nan, np.nan
    
    # Normalize the pion momentum vector
    p_pion_norm = p_pion[1:] / p_norm
    
    # Calculate cos theta for each axis
    cos_theta_r = np.dot(p_pion_norm, r_hat)
    cos_theta_n = np.dot(p_pion_norm, n_hat)
    cos_theta_k = np.dot(p_pion_norm, k_hat)
    
    # Ensure cosines are in valid range
    cos_theta_r = np.clip(cos_theta_r, -1.0, 1.0)
    cos_theta_n = np.clip(cos_theta_n, -1.0, 1.0)
    cos_theta_k = np.clip(cos_theta_k, -1.0, 1.0)
    
    return cos_theta_r, cos_theta_n, cos_theta_k

def plot_comparison_with_ratio(truth_values, reco_values, xlabel, title, bins=50, xlim=None):
    """Plot histograms with ratio plot below"""
    # Filter out NaN values
    truth_values = np.array(truth_values)
    reco_values = np.array(reco_values)
    valid_mask = ~np.isnan(truth_values) & ~np.isnan(reco_values)
    truth_values = truth_values[valid_mask]
    reco_values = reco_values[valid_mask]
    
    if len(truth_values) == 0 or len(reco_values) == 0:
        print(f"Warning: No valid data to plot for {title}")
        return
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Top Plot - Truth (filled) vs Reco (outline)
    hist_truth, bin_edges = np.histogram(truth_values, bins=bins)
    hist_reco, _ = np.histogram(reco_values, bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax[0].hist(truth_values, bins=bin_edges, alpha=0.5, color='gray', label='Truth')
    ax[0].step(bin_centers, hist_reco, where='mid', color='orange', linewidth=2, label='Reconstructed')

    ax[0].set_ylabel('Count')
    ax[0].set_title(title)
    ax[0].legend()

    # Bottom Plot - Ratio (Reco/Truth)
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
    plt.show()

def plot_relative_uncertainty(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=(-1, 1)):
    """Plot relative uncertainties between truth and reconstructed values"""
    # Calculate relative uncertainties
    rel_unc = [(reco - truth)/truth if truth != 0 else 0 
               for truth, reco in zip(truth_values, reco_values)]
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rel_unc, bins=bins, alpha=0.7)
    plt.xlabel(f'Relative Uncertainty in {component}')
    plt.ylabel('Count')
    plt.title(f'Relative Uncertainty in {component} for {particle_type}{charge}')
    plt.xlim(xlim)
    plt.grid(True)
    plt.show()
