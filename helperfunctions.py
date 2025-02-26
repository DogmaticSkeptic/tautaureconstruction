import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import scipy.optimize as opt


def define_coordinate_system(p_tau):
    """Define the {r^, n^, k^} coordinate system in the tau tau rest frame"""
    # k^ is the flight direction of tau- in the tau tau rest frame
    k_hat = p_tau[1:] / np.linalg.norm(p_tau[1:])
    
    # p^ is the direction of one of the e± beams (assume z-axis)
    p_hat = np.array([0, 0, 1])
    
    # r^ = (p^ - k^ * cosΘ) / sinΘ
    cos_theta = np.dot(k_hat, p_hat)
    sin_theta = np.sqrt(1 - cos_theta**2)
    r_hat = (p_hat - k_hat * cos_theta) / sin_theta
    
    # n^ = k^ × r^
    n_hat = np.cross(k_hat, r_hat)
    
    return r_hat, n_hat, k_hat

def compute_cos_theta(p_pion, r_hat, n_hat, k_hat):
    """Calculate cos theta for each axis in the rest frame"""
    # Normalize the pion momentum vector
    p_pion_norm = p_pion[1:] / np.linalg.norm(p_pion[1:])
    
    # Calculate cos theta for each axis
    cos_theta_r = np.dot(p_pion_norm, r_hat)
    cos_theta_n = np.dot(p_pion_norm, n_hat)
    cos_theta_k = np.dot(p_pion_norm, k_hat)
    
    return cos_theta_r, cos_theta_n, cos_theta_k

def compute_four_momentum(vec):
    """Compute four-momentum from particle vector"""
    px, py, pz, m = vec.px, vec.py, vec.pz, vec.mass
    p_magnitude = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_magnitude**2 + m**2)
    return np.array([E, px, py, pz])


# Constants
M_TAU = 1.776  # Tau mass in GeV
M_Z = 91.1876  # Z-boson mass in GeV
SIGMA_TAU = 0.05  # Tau mass uncertainty (GeV)
SIGMA_MET = 0.02822  # MET uncertainty (GeV)
SIGMA_Z = 0.0045  # Z mass uncertainty (GeV)


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

def chi_squared_nu(neutrino_params, p_pi_p, p_pi_m, MET_x, MET_y, 
                   sigma_tau=0.001, sigma_z=0.002, sigma_met=0.01):
    """Chi-squared function for neutrino momentum reconstruction"""
    p_nu_p = np.array([np.linalg.norm(neutrino_params[:3]), neutrino_params[0], neutrino_params[1], neutrino_params[2]])
    p_nu_m = np.array([np.linalg.norm(neutrino_params[3:]), neutrino_params[3], neutrino_params[4], neutrino_params[5]])

    p_tau_p = p_pi_p + p_nu_p
    p_tau_m = p_pi_m + p_nu_m

    chi2_tau_p = ((M_TAU**2 - (p_tau_p[0]**2 - np.sum(p_tau_p[1:]**2)))**2) / sigma_tau**2
    chi2_tau_m = ((M_TAU**2 - (p_tau_m[0]**2 - np.sum(p_tau_m[1:]**2)))**2) / sigma_tau**2
    chi2_MET_x = ((p_nu_p[1] + p_nu_m[1] - MET_x)**2) / sigma_met**2
    chi2_MET_y = ((p_nu_p[2] + p_nu_m[2] - MET_y)**2) / sigma_met**2
    chi2_Z_mass = ((M_Z**2 - ((p_tau_p + p_tau_m)[0]**2 - np.sum((p_tau_p + p_tau_m)[1:]**2)))**2) / sigma_z**2

    chi2_total = chi2_tau_p + chi2_tau_m + chi2_MET_x + chi2_MET_y + chi2_Z_mass

    return chi2_total


# Compute pseudorapidity function
def compute_eta(p):
    px, py, pz = p[1], p[2], p[3]
    theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    tan_theta_half = np.tan(theta / 2 + epsilon)
    # Handle cases where tan_theta_half is very small
    if abs(tan_theta_half) < epsilon:
        return np.sign(pz) * 1e10  # Return large value with correct sign
    return -np.log(abs(tan_theta_half))

# Compute phi function
def compute_phi(p):
    px, py = p[1], p[2]
    return np.arctan2(py, px)

# Compute transverse momentum function
def compute_pT(p):
    px, py = p[1], p[2]
    return np.sqrt(px**2 + py**2)

# Analysis functions
def reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y,
                                sigma_tau=0.001, sigma_z=0.002, sigma_met=0.01):
    """Reconstruct neutrino momenta using chi-squared minimization"""
    initial_guess = [MET_x / 2, MET_y / 2, 5.0, MET_x / 2, MET_y / 2, -5.0]

    # Use L-BFGS-B with bounds to prevent unphysical solutions
    bounds = [(0, None)] * 6  # All momentum components must be >= 0
    result = opt.minimize(
        chi_squared_nu, initial_guess,
        args=(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y, sigma_tau, sigma_z, sigma_met),
        method="BFGS"
    )

    p_nu_p_opt = np.array([np.linalg.norm(result.x[:3]), *result.x[:3]])
    p_nu_m_opt = np.array([np.linalg.norm(result.x[3:]), *result.x[3:]])

    return p_nu_p_opt, p_nu_m_opt, result.fun

def plot_comparison_with_ratio(truth_values, reco_values, chi2_values, xlabel, title, bins=50, xlim=None):
    """Plot histograms with ratio plot below and save to file"""
    truth_values = np.array(truth_values)
    reco_values = np.array(reco_values)
    chi2_values = np.array(chi2_values)
    
    # Filter out points with chi2 > 1e6
    valid_mask = (~np.isnan(truth_values) & 
                  ~np.isnan(reco_values) & 
                  (chi2_values <= 1e6))
    
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

def plot_2d_cos_theta_correlation(cos_theta_tau, cos_theta_pion, component, bins=50, range=(-1, 1)):
    """Plot 2D correlation between cos theta of tau and its child pion"""
    plt.figure(figsize=(8, 8))
    plt.hist2d(cos_theta_tau, cos_theta_pion, bins=bins, range=[range, range], cmap='viridis', cmin=1)
    plt.colorbar(label='Counts')
    plt.plot(range, range, 'r--', linewidth=1, label='y = x')
    plt.xlabel(f'cosθ_{component} (Tau)')
    plt.ylabel(f'cosθ_{component} (Pion)')
    plt.title(f'Correlation between Tau and Pion cosθ_{component}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/cos_theta_{component}_correlation.png')
    plt.close()

def plot_relative_uncertainty(truth_values, reco_values, chi2_values, component, particle_type, charge, bins=50, xlim=(-3, 3)):
    """Plot relative uncertainties between truth and reconstructed values and save to file"""
    
    # Filter out cases where truth value is 0 and chi2 > 1e6
    valid_data = [(t, r) for t, r, chi2 in zip(truth_values, reco_values, chi2_values) 
                 if t != 0 and chi2 <= 1e6]
    if not valid_data:
        print(f"Warning: No valid data to plot for {particle_type}{charge} {component} (all truth values are 0)")
        return
        
    rel_unc = [(r - t) / t for t, r in valid_data]

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

