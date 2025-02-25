import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import pickle
import scipy.optimize as opt


# Constants
M_TAU = 1.776  # Tau mass in GeV
M_Z = 91.1876  # Z-boson mass in GeV
SIGMA_TAU = 0.05  # Tau mass uncertainty (GeV)
SIGMA_MET = 0.02822  # MET uncertainty (GeV)
SIGMA_Z = 0.0045  # Z mass uncertainty (GeV)

NUM_CPUS = 12

def compute_four_momentum(vec):
    """Compute four-momentum from particle vector"""
    px, py, pz, m = vec.px, vec.py, vec.pz, vec.mass
    p_magnitude = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_magnitude**2 + m**2)
    return np.array([E, px, py, pz])

def plot_relative_uncertainty(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=(-3, 3)):
    """Plot relative uncertainties between truth and reconstructed values and save to file"""
    
    # Filter out cases where truth value is 0
    valid_data = [(t, r) for t, r in zip(truth_values, reco_values) if t != 0]
    if not valid_data:
        print(f"Warning: No valid data to plot for {particle_type}{charge} {component} (all truth values are 0)")
        return
        
    rel_unc = [(r - t) / t for t, r in valid_data]

    # Debug print statements
    print(f"\nDebugging {particle_type}{charge} {component}:")
    print(f"Number of events: {len(rel_unc)}")
    print(f"Number of -1 values: {sum(1 for x in rel_unc if abs(x + 1) < 1e-6)}")
    
    # Print some examples where rel_unc is close to -1
    for i, (t, r, ru) in enumerate(zip(truth_values, reco_values, rel_unc)):
        if abs(ru + 1) < 1e-6:
            print(f"\nEvent {i} with rel_unc ~ -1:")
            print(f"Truth {component}: {t}")
            print(f"Reco {component}: {r}")
            print(f"Relative uncertainty: {ru}")
            if i > 5:  # Print up to 5 examples
                break

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

def plot_momentum_comparison(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=None):
    """Plot comparison of truth vs reconstructed momentum distributions"""
    plt.figure(figsize=(10, 6))
    
    # Create bins based on data range if xlim not specified
    if xlim is None:
        all_values = truth_values + reco_values
        xlim = (min(all_values), max(all_values))
    
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    
    plt.hist(truth_values, bins=bin_edges, alpha=0.7, label='Truth')
    plt.hist(reco_values, bins=bin_edges, alpha=0.7, label='Reconstructed')
    
    plt.xlabel(f'{component} (GeV)')
    plt.ylabel('Count')
    plt.title(rf'{component} Distribution for {particle_type}{charge}')
    plt.legend()
    plt.grid(True)
    filename = f'momentum_comp_{particle_type}_{charge}_{component}.png'
    plt.savefig(f'plots/{filename}')
    plt.close()

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

    return p_nu_p_opt, p_nu_m_opt


particle_data_dict = pickle.load(open('pi_pi_recon_particles.pkl', 'rb'))

# Truth Objects
truth_tau_p = particle_data_dict['truth_tau_p']
truth_tau_m = particle_data_dict['truth_tau_m']
truth_nu_p = particle_data_dict['truth_nu_p']
truth_nu_m = particle_data_dict['truth_nu_m']
truth_pion_p = particle_data_dict['truth_tau_p_child']
truth_pion_m = particle_data_dict['truth_tau_m_child']

# Reconstructed Objects
tau_p_pion = particle_data_dict['tau_p_child1']
tau_m_pion = particle_data_dict['tau_m_child1']
MET = particle_data_dict['MET']
    
n_events = 1000

truth_data = []
reco_data = []

for i in range(n_events):
    truth_data.append((compute_four_momentum(truth_tau_p[i]), compute_four_momentum(truth_tau_m[i]),
                         compute_four_momentum(truth_pion_p[i]), compute_four_momentum(truth_pion_m[i]),
                         compute_four_momentum(truth_nu_p[i]), compute_four_momentum(truth_nu_m[i])))
    reco_data.append((compute_four_momentum(tau_p_pion[i]), compute_four_momentum(tau_m_pion[i])))

# Prepare arguments for parallel processing
reco_args = [(reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py) for i in range(n_events)]

# Reconstruct neutrino momenta using multiprocessing
def reconstruct_event(args):
    p_pi_p_reco, p_pi_m_reco, MET_x, MET_y = args
    return reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y)

# Use multiprocessing with NUM_CPUS CPUs for both methods
with Pool(processes=NUM_CPUS) as pool:
    reco_neutrino_momenta = list(tqdm(pool.imap(reconstruct_event, reco_args), total=n_events))

# Check for NaN values in reconstructed momenta
nan_count = sum(1 for momenta in reco_neutrino_momenta 
                if any(np.isnan(p).any() for p in momenta))
print(f"Found {nan_count} events with NaN values in reconstructed neutrino momenta")


for component, idx in [('px', 1), ('py', 2), ('pz', 3)]:
    # Neutrino+
    truth_p = [truth_data[i][4][idx] for i in range(n_events)]
    reco_p = [reco_neutrino_momenta[i][0][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_p, reco_p, component, 'Neutrino', '+')
    plot_momentum_comparison(truth_p, reco_p, component, 'Neutrino', '+')

    # Neutrino-
    truth_m = [truth_data[i][5][idx] for i in range(n_events)]
    reco_m = [reco_neutrino_momenta[i][1][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_m, reco_m, component, 'Neutrino', '-')
    plot_momentum_comparison(truth_m, reco_m, component, 'Neutrino', '-')
