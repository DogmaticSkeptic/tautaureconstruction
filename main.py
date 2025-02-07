import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import pickle
NUM_CPUS = 4
from helperfunctions import (
    boost_to_rest_frame, define_coordinate_system,
    reconstruct_neutrino_momenta, plot_comparison_with_ratio, plot_relative_uncertainty,
    compute_cos_theta, chi_squared_nu, compute_four_momentum, compute_eta, compute_phi, compute_pT
)


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
    
n_events = 100

truth_data = []
reco_data = []

for i in range(n_events):
    truth_data.append((compute_four_momentum(truth_tau_p[i]), compute_four_momentum(truth_tau_m[i]),
                         compute_four_momentum(truth_pion_p[i]), compute_four_momentum(truth_pion_m[i]),
                         compute_four_momentum(truth_nu_p[i]), compute_four_momentum(truth_nu_m[i])))
    reco_data.append((compute_four_momentum(tau_p_pion[i]), compute_four_momentum(tau_m_pion[i])))

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


# Prepare arguments for parallel processing
reco_args = [(reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py) for i in range(n_events)]

# Reconstruct neutrino momenta using multiprocessing
def reconstruct_event(args):
    p_pi_p_reco, p_pi_m_reco, MET_x, MET_y = args
    return reconstruct_neutrino_momenta(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y)

# Use multiprocessing with NUM_CPUS CPUs
with Pool(processes=NUM_CPUS) as pool:
    reco_neutrino_momenta = list(tqdm(pool.imap(reconstruct_event, reco_args), total=n_events))

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
                          title=r'Truth vs. Reconstructed Neutrino+ Pseudorapidity',
                          xlim=(-3.5, 3.5))
plot_comparison_with_ratio(truth_eta_m, reco_eta_m, xlabel=r'Pseudorapidity $\eta$',
                          title=r'Truth vs. Reconstructed Neutrino- Pseudorapidity',
                          xlim=(-3.5, 3.5))

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
                          title='Truth vs. Reconstructed Tau+ Pseudorapidity',
                          xlim=(-3.5, 3.5))
plot_comparison_with_ratio(truth_tau_eta_m, reco_tau_eta_m, xlabel=r'Pseudorapidity $\eta$', 
                          title='Truth vs. Reconstructed Tau- Pseudorapidity',
                          xlim=(-3.5, 3.5))

# Plot phi for positive and negative taus
plot_comparison_with_ratio(truth_tau_phi_p, reco_tau_phi_p, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Tau+ $\phi$')
plot_comparison_with_ratio(truth_tau_phi_m, reco_tau_phi_m, xlabel=r'Azimuthal Angle $\phi$ (radians)', 
                          title='Truth vs. Reconstructed Tau- $\phi$')

# Plot transverse momentum (pT) for positive and negative taus
plot_comparison_with_ratio(truth_tau_pT_p, reco_tau_pT_p, xlabel=r'Transverse Momentum $p_T$ (GeV)',
                          title=r'Truth vs. Reconstructed Tau+ Transverse Momentum',
                          bins=50, xlim=(0, 100))
plot_comparison_with_ratio(truth_tau_pT_m, reco_tau_pT_m, xlabel=r'Transverse Momentum $p_T$ (GeV)',
                          title=r'Truth vs. Reconstructed Tau- Transverse Momentum',
                          bins=50, xlim=(0, 100))

# Initialize lists to store cos theta values
truth_cos_theta_r_p, truth_cos_theta_n_p, truth_cos_theta_k_p = [], [], []
truth_cos_theta_r_m, truth_cos_theta_n_m, truth_cos_theta_k_m = [], [], []
reco_cos_theta_r_p, reco_cos_theta_n_p, reco_cos_theta_k_p = [], [], []
reco_cos_theta_r_m, reco_cos_theta_n_m, reco_cos_theta_k_m = [], [], []

for i in range(n_events):
    # Get tau-tau system momentum for truth and reco
    p_tau_tau_truth = truth_data[i][0] + truth_data[i][1]
    p_tau_tau_reco = reco_tau_momenta[i][0] + reco_tau_momenta[i][1]
    
    # Boost truth particles to tau-tau rest frame
    p_tau_p_truth_rest = boost_to_rest_frame(truth_data[i][0], p_tau_tau_truth)
    p_tau_m_truth_rest = boost_to_rest_frame(truth_data[i][1], p_tau_tau_truth)
    p_pion_p_truth_rest = boost_to_rest_frame(truth_data[i][2], p_tau_tau_truth)
    p_pion_m_truth_rest = boost_to_rest_frame(truth_data[i][3], p_tau_tau_truth)
    
    # Define coordinate system in tau-tau rest frame
    r_hat_tautau, n_hat_tautau, k_hat_tautau = define_coordinate_system(p_tau_m_truth_rest)

    # Boost pion+ to single tau+ rest frame
    p_pion_p_single_rest = boost_to_rest_frame(p_pion_p_truth_rest, p_tau_p_truth_rest)

    # Calculate cosθ in single tau+ frame using tau-tau rest frame basis
    truth_cos_theta_r_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[0])
    truth_cos_theta_n_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[1])
    truth_cos_theta_k_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[2])

    # Boost to single tau- rest frame
    p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_truth_rest, p_tau_m_truth_rest)

    # Calculate cosθ in single tau- frame using tau-tau rest frame basis
    truth_cos_theta_r_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[0])
    truth_cos_theta_n_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[1])
    truth_cos_theta_k_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_tautau, n_hat_tautau, k_hat_tautau)[2])

    # Boost reco particles to tau-tau rest frame
    p_tau_p_reco_rest = boost_to_rest_frame(reco_tau_momenta[i][0], p_tau_tau_reco)
    p_tau_m_reco_rest = boost_to_rest_frame(reco_tau_momenta[i][1], p_tau_tau_reco)
    p_pion_p_reco_rest = boost_to_rest_frame(reco_data[i][0], p_tau_tau_reco)
    p_pion_m_reco_rest = boost_to_rest_frame(reco_data[i][1], p_tau_tau_reco)
    
    # Define coordinate system in tau-tau rest frame
    r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau = define_coordinate_system(p_tau_m_reco_rest)

    # Boost to single tau+ rest frame
    p_pion_p_single_rest = boost_to_rest_frame(p_pion_p_reco_rest, p_tau_p_reco_rest)

    # Calculate cosθ in single tau+ frame using tau-tau rest frame basis
    reco_cos_theta_r_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[0])
    reco_cos_theta_n_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[1])
    reco_cos_theta_k_p.append(compute_cos_theta(p_pion_p_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[2])

    # Boost to single tau- rest frame
    p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_reco_rest, p_tau_m_reco_rest)

    # Calculate cosθ in single tau- frame using tau-tau rest frame basis
    reco_cos_theta_r_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[0])
    reco_cos_theta_n_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[1])
    reco_cos_theta_k_m.append(compute_cos_theta(p_pion_m_single_rest, r_hat_reco_tautau, n_hat_reco_tautau, k_hat_reco_tautau)[2])

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

# Calculate the Cij matrix elements using binned data
C = np.zeros((3, 3))
bins = 50  # Number of bins for histogramming
bin_range = (-1, 1)  # Range of cos theta values

# Calculate each matrix element
for i in range(3):
    for j in range(3):
        # Get the corresponding cos theta values from reconstructed data
        cos_theta_p = [reco_cos_theta_r_p, reco_cos_theta_n_p, reco_cos_theta_k_p][i]
        cos_theta_m = [reco_cos_theta_r_m, reco_cos_theta_n_m, reco_cos_theta_k_m][j]
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(cos_theta_p, cos_theta_m, 
                                            bins=bins, range=[bin_range, bin_range])
        
        # Calculate bin centers
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        
        # Calculate bin width
        bin_width = xedges[1] - xedges[0]
        
        # Calculate expectation value by multiplying bin contents and bin width
        expectation = np.sum(xcenters * ycenters * hist) * bin_width / np.sum(hist)
        
        # Fill the matrix element
        C[i, j] = -9 * expectation

# Calculate eigenvalues and sort them
eigenvalues = np.linalg.eigvals(C)
sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

# Calculate the sum of the two largest eigenvalues
concurrence = np.sum(sorted_eigenvalues[:2])

# Print the results
print("\nConcurrence Analysis:")
print("Cij Matrix:")
print(C)
print("\nEigenvalues:", sorted_eigenvalues)
print("Sum of two largest eigenvalues (Concurrence):", concurrence)

# Save the results to a file
with open('concurrence_results.txt', 'w') as f:
    f.write("Concurrence Analysis Results\n")
    f.write("===========================\n\n")
    f.write("Cij Matrix:\n")
    f.write(str(C) + "\n\n")
    f.write("Eigenvalues: " + str(sorted_eigenvalues) + "\n")
    f.write("Sum of two largest eigenvalues (Concurrence): " + str(concurrence) + "\n")

