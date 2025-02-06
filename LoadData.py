#!/usr/bin/env python3
import pickle
import numpy as np
from data_loading.load_data import compute_four_momentum
import matplotlib.pyplot as plt
from tqdm import tqdm
from helperfunctions import (
    boost_to_rest_frame, boost_three_vector, define_coordinate_system,
    reconstruct_neutrino_momenta, plot_comparison_with_ratio, plot_relative_uncertainty,
    compute_cos_theta, chi_squared_nu
)


particle_data_dict = pickle.load(open('pi_pi_recon_particles.pkl', 'rb'))

#print(particle_data_dict.keys())


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
    
n_events = 100

truth_data = []
reco_data = []

print(len(truth_tau_p))

for i in range(n_events):
    truth_data.append((compute_four_momentum(truth_tau_p[i]), compute_four_momentum(truth_tau_m[i]),
                         compute_four_momentum(truth_pion_p[i]), compute_four_momentum(truth_pion_m[i]),
                         compute_four_momentum(truth_nu_p[i]), compute_four_momentum(truth_nu_m[i])))
    reco_data.append((compute_four_momentum(tau_p_pion[i]), compute_four_momentum(tau_m_pion[i])))


# Plot results
true_energies = [p[2][0] for p in tqdm(truth_data)] + [p[3][0] for p in tqdm(truth_data)]
reco_energies = [p[0][0] for p in tqdm(reco_data)] + [p[1][0] for p in tqdm(reco_data)]

# Define x-axis range
x_min, x_max = 0, 200  # Adjust this range as needed

# Define number of bins and generate bin edges within x_min and x_max
num_bins = 10
bins = np.linspace(x_min, x_max, num_bins + 1)

# Plot histograms using the defined bins
#plt.hist(true_energies, bins=bins, alpha=0.7, label="True Pion Energies")
#plt.hist(reco_energies, bins=bins, alpha=0.7, label="Reco Pion Energies")

# Labels and title
#plt.xlabel("Energy (GeV)")
#plt.ylabel("Count")
#plt.legend()
#plt.title("True vs. Reconstructed Pion Energies")

# Set the x-axis range explicitly (though the bins already enforce it)
#plt.xlim(x_min, x_max)

#plt.show()

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


# Reconstruct neutrino momenta for all events
reco_neutrino_momenta = []

for i in tqdm(range(n_events)):
    # Extract reconstructed pion momenta
    p_pi_p_reco = reco_data[i][0]
    p_pi_m_reco = reco_data[i][1]

    # Extract MET components
    MET_x = MET[i].px
    MET_y = MET[i].py

    # Perform reconstruction
    p_nu_p_opt, p_nu_m_opt = reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y)
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

# Plot pseudorapidity (eta) for positive and negative neutrinos
plot_comparison_with_ratio(truth_eta_p, reco_eta_p, xlabel=r'Pseudorapidity $\eta$',
                          title=r'Truth vs. Reconstructed Neutrino+ Pseudorapidity')
plot_comparison_with_ratio(truth_eta_m, reco_eta_m, xlabel=r'Pseudorapidity $\eta$',
                          title=r'Truth vs. Reconstructed Neutrino- Pseudorapidity')

# Plot phi for positive and negative neutrinos
plot_comparison_with_ratio(truth_phi_p, reco_phi_p, xlabel=r'Azimuthal Angle $\phi$ (radians)',
                          title=r'Truth vs. Reconstructed Neutrino+ $\phi$')
plot_comparison_with_ratio(truth_phi_m, reco_phi_m, xlabel=r'Azimuthal Angle $\phi$ (radians)',
                          title=r'Truth vs. Reconstructed Neutrino- $\phi$')

# Plot transverse momentum (pT) for positive and negative neutrinos
plot_comparison_with_ratio(truth_pT_p, reco_pT_p, xlabel=r'Transverse Momentum $p_T$ (GeV)',
                          title=r'Truth vs. Reconstructed Neutrino+ Transverse Momentum',
                          bins=50, xlim=(0, 100))
plot_comparison_with_ratio(truth_pT_m, reco_pT_m, xlabel=r'Transverse Momentum $p_T$ (GeV)',
                          title=r'Truth vs. Reconstructed Neutrino- Transverse Momentum',
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
                          title=r'Truth vs. Reconstructed Tau+ Transverse Momentum',
                          bins=50, xlim=(0, 150))
plot_comparison_with_ratio(truth_tau_pT_m, reco_tau_pT_m, xlabel=r'Transverse Momentum $p_T$ (GeV)',
                          title=r'Truth vs. Reconstructed Tau- Transverse Momentum',
                          bins=50, xlim=(0, 150))

# Initialize lists to store cos theta values
truth_cos_theta_r_p, truth_cos_theta_n_p, truth_cos_theta_k_p = [], [], []
truth_cos_theta_r_m, truth_cos_theta_n_m, truth_cos_theta_k_m = [], [], []
reco_cos_theta_r_p, reco_cos_theta_n_p, reco_cos_theta_k_p = [], [], []
reco_cos_theta_r_m, reco_cos_theta_n_m, reco_cos_theta_k_m = [], [], []

# Debug flags
DEBUG_BOOST = True  # Set to True to print boost debug info

for i in range(n_events):
    # Get tau-tau system momentum for truth and reco
    p_tau_tau_truth = truth_data[i][0] + truth_data[i][1]
    p_tau_tau_reco = reco_tau_momenta[i][0] + reco_tau_momenta[i][1]
    
    if DEBUG_BOOST:
        print(f"\nEvent {i}:")
        print(f"Initial tau+ momentum: {truth_data[i][0]}")
        print(f"Initial tau- momentum: {truth_data[i][1]}")
    
    # Boost truth particles to tau-tau rest frame
    p_tau_p_truth_rest = boost_to_rest_frame(truth_data[i][0], p_tau_tau_truth, debug=DEBUG_BOOST)
    p_tau_m_truth_rest = boost_to_rest_frame(truth_data[i][1], p_tau_tau_truth)
    
    # Verify we're in the tau-tau rest frame
    if DEBUG_BOOST:
        p_tau_tau_rest = p_tau_p_truth_rest + p_tau_m_truth_rest
        print(f"Total momentum in tau-tau rest frame: {np.linalg.norm(p_tau_tau_rest[1:]):.4f} GeV")
    p_pion_p_truth_rest = boost_to_rest_frame(truth_data[i][2], p_tau_tau_truth)
    p_pion_m_truth_rest = boost_to_rest_frame(truth_data[i][3], p_tau_tau_truth)
    
    # Define coordinate system in tau-tau rest frame
    r_hat_tautau, n_hat_tautau, k_hat_tautau = define_coordinate_system(p_tau_p_truth_rest, p_tau_m_truth_rest)
    
    # Boost to single tau+ rest frame
    p_tau_p_single_rest = boost_to_rest_frame(p_tau_p_truth_rest, p_tau_p_truth_rest)
    p_pion_p_single_rest = boost_to_rest_frame(p_pion_p_truth_rest, p_tau_p_truth_rest)
    
    # Boost coordinate axes to single tau+ frame
    r_hat_single_p = boost_three_vector(r_hat_tautau, p_tau_p_truth_rest)
    n_hat_single_p = boost_three_vector(n_hat_tautau, p_tau_p_truth_rest)
    k_hat_single_p = boost_three_vector(k_hat_tautau, p_tau_p_truth_rest)
    
    # Normalize boosted axes
    r_hat_single_p /= np.linalg.norm(r_hat_single_p)
    n_hat_single_p /= np.linalg.norm(n_hat_single_p)
    k_hat_single_p /= np.linalg.norm(k_hat_single_p)
    
    # Calculate cosθ in single tau+ frame
    cos_theta_r_p, cos_theta_n_p, cos_theta_k_p = compute_cos_theta(
        p_pion_p_single_rest,
        r_hat_single_p, n_hat_single_p, k_hat_single_p
    )
    truth_cos_theta_r_p.append(cos_theta_r_p)
    truth_cos_theta_n_p.append(cos_theta_n_p)
    truth_cos_theta_k_p.append(cos_theta_k_p)
    
    # Boost to single tau- rest frame
    p_tau_m_single_rest = boost_to_rest_frame(p_tau_m_truth_rest, p_tau_m_truth_rest)
    p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_truth_rest, p_tau_m_truth_rest)
    
    # Boost coordinate axes to single tau- frame
    r_hat_single_m = boost_three_vector(r_hat_tautau, p_tau_m_truth_rest)
    n_hat_single_m = boost_three_vector(n_hat_tautau, p_tau_m_truth_rest)
    k_hat_single_m = boost_three_vector(k_hat_tautau, p_tau_m_truth_rest)
    
    # Normalize boosted axes
    r_hat_single_m /= np.linalg.norm(r_hat_single_m)
    n_hat_single_m /= np.linalg.norm(n_hat_single_m)
    k_hat_single_m /= np.linalg.norm(k_hat_single_m)
    
    # Calculate cosθ in single tau- frame
    cos_theta_r_m, cos_theta_n_m, cos_theta_k_m = compute_cos_theta(
        p_pion_m_single_rest,
        r_hat_single_m, n_hat_single_m, k_hat_single_m
    )
    truth_cos_theta_r_m.append(cos_theta_r_m)
    truth_cos_theta_n_m.append(cos_theta_n_m)
    truth_cos_theta_k_m.append(cos_theta_k_m)
    
    # Boost reco particles to tau-tau rest frame
    p_tau_p_reco_rest = boost_to_rest_frame(reco_tau_momenta[i][0], p_tau_tau_reco)
    p_tau_m_reco_rest = boost_to_rest_frame(reco_tau_momenta[i][1], p_tau_tau_reco)
    p_pion_p_reco_rest = boost_to_rest_frame(reco_data[i][0], p_tau_tau_reco)
    p_pion_m_reco_rest = boost_to_rest_frame(reco_data[i][1], p_tau_tau_reco)
    
    # Define coordinate system in tau-tau rest frame
    r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau = define_coordinate_system(p_tau_p_reco_rest, p_tau_m_reco_rest)
    
    # Boost to single tau+ rest frame
    p_tau_p_single_rest = boost_to_rest_frame(p_tau_p_reco_rest, p_tau_p_reco_rest)
    p_pion_p_single_rest = boost_to_rest_frame(p_pion_p_reco_rest, p_tau_p_reco_rest)
    
    # Boost coordinate axes to single tau+ frame
    r_hat_single_p = boost_three_vector(r_hat_reco_tautau, p_tau_p_reco_rest)
    n_hat_single_p = boost_three_vector(n_hat_reco_tautau, p_tau_p_reco_rest)
    k_hat_single_p = boost_three_vector(k_hat_reco_tautau, p_tau_p_reco_rest)
    
    # Normalize boosted axes
    r_hat_single_p /= np.linalg.norm(r_hat_single_p)
    n_hat_single_p /= np.linalg.norm(n_hat_single_p)
    k_hat_single_p /= np.linalg.norm(k_hat_single_p)
    
    # Calculate cosθ in single tau+ frame
    cos_theta_r_p, cos_theta_n_p, cos_theta_k_p = compute_cos_theta(
        p_pion_p_single_rest,
        r_hat_single_p, n_hat_single_p, k_hat_single_p
    )
    reco_cos_theta_r_p.append(cos_theta_r_p)
    reco_cos_theta_n_p.append(cos_theta_n_p)
    reco_cos_theta_k_p.append(cos_theta_k_p)
    
    # Boost to single tau- rest frame
    p_tau_m_single_rest = boost_to_rest_frame(p_tau_m_reco_rest, p_tau_m_reco_rest)
    p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_reco_rest, p_tau_m_reco_rest)
    
    # Boost coordinate axes to single tau- frame
    r_hat_single_m = boost_three_vector(r_hat_reco_tautau, p_tau_m_reco_rest)
    n_hat_single_m = boost_three_vector(n_hat_reco_tautau, p_tau_m_reco_rest)
    k_hat_single_m = boost_three_vector(k_hat_reco_tautau, p_tau_m_reco_rest)
    
    # Normalize boosted axes
    r_hat_single_m /= np.linalg.norm(r_hat_single_m)
    n_hat_single_m /= np.linalg.norm(n_hat_single_m)
    k_hat_single_m /= np.linalg.norm(k_hat_single_m)
    
    # Calculate cosθ in single tau- frame
    cos_theta_r_m, cos_theta_n_m, cos_theta_k_m = compute_cos_theta(
        p_pion_m_single_rest,
        r_hat_single_m, n_hat_single_m, k_hat_single_m
    )
    reco_cos_theta_r_m.append(cos_theta_r_m)
    reco_cos_theta_n_m.append(cos_theta_n_m)
    reco_cos_theta_k_m.append(cos_theta_k_m)

# Plot cos theta for tau+
plot_comparison_with_ratio(truth_cos_theta_r_p, reco_cos_theta_r_p, xlabel=r'$\cos \theta_r$', 
                          title='Truth vs. Reconstructed $\cos \theta_r$ for Tau+')
plot_comparison_with_ratio(truth_cos_theta_n_p, reco_cos_theta_n_p, xlabel=r'$\cos \theta_n$', 
                          title='Truth vs. Reconstructed $\cos \theta_n$ for Tau+')
plot_comparison_with_ratio(truth_cos_theta_k_p, reco_cos_theta_k_p, xlabel=r'$\cos \theta_k$', 
                          title='Truth vs. Reconstructed $\cos \theta_k$ for Tau+')

# Plot cos theta for tau-
plot_comparison_with_ratio(truth_cos_theta_r_m, reco_cos_theta_r_m, xlabel=r'$\cos \theta_r$',
                          title=r'Truth vs. Reconstructed $\cos \theta_r$ for Tau-')
plot_comparison_with_ratio(truth_cos_theta_n_m, reco_cos_theta_n_m, xlabel=r'$\cos \theta_n$',
                          title=r'Truth vs. Reconstructed $\cos \theta_n$ for Tau-')
plot_comparison_with_ratio(truth_cos_theta_k_m, reco_cos_theta_k_m, xlabel=r'$\cos \theta_k$',
                          title=r'Truth vs. Reconstructed $\cos \theta_k$ for Tau-')

# Function to plot relative uncertainties
def plot_relative_uncertainty(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=(-1, 1)):
    # Calculate relative uncertainties
    rel_unc = [(reco - truth)/truth if truth != 0 else 0
               for truth, reco in zip(truth_values, reco_values)]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rel_unc, bins=bins, alpha=0.7)
    plt.xlabel(f'Relative Uncertainty in {component}')
    plt.ylabel('Count')
    plt.title(rf'Relative Uncertainty in {component} for {particle_type}{charge}')
    plt.xlim(xlim)
    plt.grid(True)
    plt.show()

# Plot relative uncertainties for tau components
for component, idx in [('px', 1), ('py', 2), ('pz', 3)]:
    # Tau+
    truth_p = [truth_data[i][0][idx] for i in range(n_events)]
    reco_p = [reco_tau_momenta[i][0][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_p, reco_p, component, 'Tau', '+')

    # Tau-
    truth_m = [truth_data[i][1][idx] for i in range(n_events)]
    reco_m = [reco_tau_momenta[i][1][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_m, reco_m, component, 'Tau', '-')


# Plot relative uncertainties for neutrino components
for component, idx in [('px', 1), ('py', 2), ('pz', 3)]:
    # Neutrino+
    truth_p = [truth_data[i][4][idx] for i in range(n_events)]
    reco_p = [reco_neutrino_momenta[i][0][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_p, reco_p, component, 'Neutrino', '+')

    # Neutrino-
    truth_m = [truth_data[i][5][idx] for i in range(n_events)]
    reco_m = [reco_neutrino_momenta[i][1][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_m, reco_m, component, 'Neutrino', '-')


test = boost_to_rest_frame(p_tau_tau_truth, p_tau_tau_truth)
print(test)
