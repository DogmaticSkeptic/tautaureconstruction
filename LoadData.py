import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.optimize as opt
particle_data_dict = pickle.load(open('pi_pi_recon_particles.pkl', 'rb'))

print(particle_data_dict.keys())


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

def compute_four_momentum(vec):
    px, py, pz, m = vec.px, vec.py, vec.pz, vec.mass
    p_magnitude = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_magnitude**2 + m**2)  # Using natural units (c=1)
        
    return np.array([E, px, py, pz])
    
n_events = 100

truth_data = []
reco_data = []

print(len(truth_tau_p))

for i in range(n_events):
    truth_data.append((compute_four_momentum(truth_tau_p[i]), compute_four_momentum(truth_tau_m[i]), compute_four_momentum(truth_pion_p[i]), compute_four_momentum(truth_pion_m[i]), compute_four_momentum(truth_nu_p[i]), compute_four_momentum(truth_nu_m[i])))
    reco_data.append((compute_four_momentum(tau_p_pion[i]), compute_four_momentum(tau_m_pion[i])))

#print(tau_p_pion[-1].mass)

# Plot results
true_energies = [p[2][0] for p in tqdm(truth_data)] + [p[3][0] for p in tqdm(truth_data)]
reco_energies = [p[0][0] for p in tqdm(reco_data)] + [p[1][0] for p in tqdm(reco_data)]

# Define x-axis range
x_min, x_max = 0, 200  # Adjust this range as needed

# Define number of bins and generate bin edges within x_min and x_max
num_bins = 10
bins = np.linspace(x_min, x_max, num_bins + 1)

# Plot histograms using the defined bins
plt.hist(true_energies, bins=bins, alpha=0.7, label="True Pion Energies")
plt.hist(reco_energies, bins=bins, alpha=0.7, label="Reco Pion Energies")

# Labels and title
plt.xlabel("Energy (GeV)")
plt.ylabel("Count")
plt.legend()
plt.title("True vs. Reconstructed Pion Energies")

# Set the x-axis range explicitly (though the bins already enforce it)
plt.xlim(x_min, x_max)

plt.show()

# Extract neutrino data for plotting
true_neutrino_energies = [p[4][0] for p in truth_data] + [p[5][0] for p in truth_data]
true_neutrino_eta = []
true_neutrino_phi = []

# Calculate pseudorapidity (eta) and phi for neutrinos
for event in truth_data:
    for i in [4, 5]:  # Neutrinos
        px, py, pz = event[i][1:]
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)  # Calculate theta
        eta = -np.log(np.tan(theta / 2))  # Convert to pseudorapidity
        phi = np.arctan2(py, px) % (2 * np.pi)  # Keep phi in [0, 2π]

        true_neutrino_eta.append(eta)
        true_neutrino_phi.append(phi)

# Plot neutrino energy distribution
plt.figure(figsize=(10, 6))
plt.hist(true_neutrino_energies, bins=50, alpha=0.7, color='blue', label='Neutrino Energies')
plt.xlabel('Energy (GeV)')
plt.ylabel('Count')
plt.title('Neutrino Energy Distribution')
plt.legend()
plt.show()

# Plot neutrino pseudorapidity distribution
plt.figure(figsize=(10, 6))
plt.hist(true_neutrino_eta, bins=50, alpha=0.7, color='blue', label='Neutrino Pseudorapidity')
plt.xlabel(r'$\eta$')
plt.ylabel('Count')
plt.title('Neutrino Pseudorapidity Distribution')
plt.legend()
plt.show()

# Plot neutrino phi distribution
plt.figure(figsize=(10, 6))
plt.hist(true_neutrino_phi, bins=50, alpha=0.7, color='blue', label='Neutrino Phi')
plt.xlabel(r'$\phi$ (radians)')
plt.ylabel('Count')
plt.title('Neutrino Phi Distribution')
plt.legend()
plt.show()


# Constants
m_tau = 1.776  # Tau mass in GeV
m_Z = 91.1876  # Z-boson mass in GeV
sigma_tau = 0.001  # Tau mass uncertainty (GeV)
sigma_MET = 0.01  # MET uncertainty (GeV)
sigma_Z = 0.002  # Z mass uncertainty (GeV)

# Chi-squared function to optimize neutrino momenta
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

# Reconstruct neutrino momenta for all events
reco_neutrino_momenta = []


###################### DOING THE RECONSTRUCTION
for i in tqdm(range(n_events)):
    # Extract reconstructed pion momenta
    p_pi_p_reco = reco_data[i][0]
    p_pi_m_reco = reco_data[i][1]

    # Extract MET components
    MET_x = MET[i].px
    MET_y = MET[i].py

    # Initial guess: Set neutrino momenta to half MET for transverse components and 5 GeV in z
    initial_guess = [MET_x/2, MET_y/2, 5.0, MET_x/2, MET_y/2, -5.0]

    # Minimize chi-squared
    result = opt.minimize(chi_squared_nu, initial_guess, args=(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y), method='BFGS')

    # Extract optimized neutrino momenta
    p_nu_p_opt = np.array([np.linalg.norm(result.x[:3]), *result.x[:3]])
    p_nu_m_opt = np.array([np.linalg.norm(result.x[3:]), *result.x[3:]])

    reco_neutrino_momenta.append((p_nu_p_opt, p_nu_m_opt))


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

# Collect truth and reconstructed values separated by charge
truth_eta_p = []
truth_phi_p = []
truth_pT_p = []
reco_eta_p = []
reco_phi_p = []
reco_pT_p = []

truth_eta_m = []
truth_phi_m = []
truth_pT_m = []
reco_eta_m = []
reco_phi_m = []
reco_pT_m = []

for i in range(n_events):
    # Truth neutrinos
    truth_eta_p.append(compute_eta(truth_data[i][4]))  # nu_p
    truth_phi_p.append(compute_phi(truth_data[i][4]))
    truth_pT_p.append(compute_pT(truth_data[i][4]))
    
    truth_eta_m.append(compute_eta(truth_data[i][5]))  # nu_m
    truth_phi_m.append(compute_phi(truth_data[i][5]))
    truth_pT_m.append(compute_pT(truth_data[i][5]))

    # Reconstructed neutrinos
    reco_eta_p.append(compute_eta(reco_neutrino_momenta[i][0]))  # nu_p
    reco_phi_p.append(compute_phi(reco_neutrino_momenta[i][0]))
    reco_pT_p.append(compute_pT(reco_neutrino_momenta[i][0]))
    
    reco_eta_m.append(compute_eta(reco_neutrino_momenta[i][1]))  # nu_m
    reco_phi_m.append(compute_phi(reco_neutrino_momenta[i][1]))
    reco_pT_m.append(compute_pT(reco_neutrino_momenta[i][1]))

# Function to plot histograms with ratio (without normalization on the top)
def plot_comparison_with_ratio(truth_values, reco_values, xlabel, title, bins=50, xlim=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Top Plot - Truth (filled) vs Reco (outline)
    hist_truth, bin_edges = np.histogram(truth_values, bins=bins)
    hist_reco, _ = np.histogram(reco_values, bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax[0].hist(truth_values, bins=bin_edges, alpha=0.5, color='gray', label='Truth')  # No normalization
    ax[0].step(bin_centers, hist_reco, where='mid', color='orange', linewidth=2, label='Reconstructed')

    ax[0].set_ylabel('Count')
    ax[0].set_title(title)
    ax[0].legend()

    # Bottom Plot - Ratio (Reco/Truth)
    ratio = np.divide(hist_reco, hist_truth, out=np.zeros_like(hist_reco, dtype=float), where=hist_truth > 0)

    ax[1].plot(bin_centers, ratio, 'o-', color='black', markersize=4)
    ax[1].axhline(1.0, linestyle='--', color='red', linewidth=1)  # Reference line at 1
    ax[1].set_ylim(0, 2)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Reco / Truth')

    if xlim is not None:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)
    plt.tight_layout()
    plt.show()

# Plot pseudorapidity (eta) for positive and negative neutrinos
plot_comparison_with_ratio(truth_eta_p, reco_eta_p, xlabel=r'Pseudorapidity $\eta$', 
                          title='Truth vs. Reconstructed Neutrino+ Pseudorapidity')
plot_comparison_with_ratio(truth_eta_m, reco_eta_m, xlabel=r'Pseudorapidity $\eta$', 
                          title='Truth vs. Reconstructed Neutrino- Pseudorapidity')

# Plot phi for positive and negative neutrinos
plot_comparison_with_ratio(truth_phi_p, reco_phi_p, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Neutrino+ $\phi$')
plot_comparison_with_ratio(truth_phi_m, reco_phi_m, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Neutrino- $\phi$')

# Plot transverse momentum (pT) for positive and negative neutrinos
plot_comparison_with_ratio(truth_pT_p, reco_pT_p, xlabel=r'Transverse Momentum $p_T$ (GeV)', 
                          title='Truth vs. Reconstructed Neutrino+ Transverse Momentum',
                          bins=50, xlim=(0, 100))
plot_comparison_with_ratio(truth_pT_m, reco_pT_m, xlabel=r'Transverse Momentum $p_T$ (GeV)', 
                          title='Truth vs. Reconstructed Neutrino- Transverse Momentum',
                          bins=50, xlim=(0, 100))


# Compute reconstructed tau momenta from reconstructed neutrinos and pions
reco_tau_momenta = []

for i in range(n_events):
    # Truth tau momenta
    p_tau_p_truth = truth_data[i][0]
    p_tau_m_truth = truth_data[i][1]

    # Reconstructed tau momenta
    p_tau_p_reco = reco_data[i][0] + reco_neutrino_momenta[i][0]
    p_tau_m_reco = reco_data[i][1] + reco_neutrino_momenta[i][1]

    reco_tau_momenta.append((p_tau_p_reco, p_tau_m_reco))

# Collect truth and reconstructed tau properties separated by charge
truth_tau_eta_p = []
truth_tau_phi_p = []
truth_tau_pT_p = []
reco_tau_eta_p = []
reco_tau_phi_p = []
reco_tau_pT_p = []

truth_tau_eta_m = []
truth_tau_phi_m = []
truth_tau_pT_m = []
reco_tau_eta_m = []
reco_tau_phi_m = []
reco_tau_pT_m = []

for i in range(n_events):
    # Truth taus
    truth_tau_eta_p.append(compute_eta(truth_data[i][0]))  # tau+
    truth_tau_phi_p.append(compute_phi(truth_data[i][0]))
    truth_tau_pT_p.append(compute_pT(truth_data[i][0]))
    
    truth_tau_eta_m.append(compute_eta(truth_data[i][1]))  # tau-
    truth_tau_phi_m.append(compute_phi(truth_data[i][1]))
    truth_tau_pT_m.append(compute_pT(truth_data[i][1]))

    # Reconstructed taus
    reco_tau_eta_p.append(compute_eta(reco_tau_momenta[i][0]))  # tau+
    reco_tau_phi_p.append(compute_phi(reco_tau_momenta[i][0]))
    reco_tau_pT_p.append(compute_pT(reco_tau_momenta[i][0]))
    
    reco_tau_eta_m.append(compute_eta(reco_tau_momenta[i][1]))  # tau-
    reco_tau_phi_m.append(compute_phi(reco_tau_momenta[i][1]))
    reco_tau_pT_m.append(compute_pT(reco_tau_momenta[i][1]))

# Plot pseudorapidity (eta) for positive and negative taus
plot_comparison_with_ratio(truth_tau_eta_p, reco_tau_eta_p, xlabel=r'Pseudorapidity $\eta$', 
                          title='Truth vs. Reconstructed Tau+ Pseudorapidity')
plot_comparison_with_ratio(truth_tau_eta_m, reco_tau_eta_m, xlabel=r'Pseudorapidity $\eta$', 
                          title='Truth vs. Reconstructed Tau- Pseudorapidity')

# Plot phi for positive and negative taus
plot_comparison_with_ratio(truth_tau_phi_p, reco_tau_phi_p, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Tau+ $\phi$')
plot_comparison_with_ratio(truth_tau_phi_m, reco_tau_phi_m, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Tau- $\phi$')

# Plot transverse momentum (pT) for positive and negative taus
plot_comparison_with_ratio(truth_tau_pT_p, reco_tau_pT_p, xlabel=r'Transverse Momentum $p_T$ (GeV)', 
                          title='Truth vs. Reconstructed Tau+ Transverse Momentum',
                          bins=50, xlim=(0, 150))
plot_comparison_with_ratio(truth_tau_pT_m, reco_tau_pT_m, xlabel=r'Transverse Momentum $p_T$ (GeV)', 
                          title='Truth vs. Reconstructed Tau- Transverse Momentum',
                          bins=50, xlim=(0, 150))
