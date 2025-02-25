import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import pickle
NUM_CPUS = 12
from helperfunctions import define_coordinate_system, boost_to_rest_frame, compute_cos_theta, reconstruct_neutrino_momenta, plot_comparison_with_ratio, plot_relative_uncertainty,compute_four_momentum, compute_eta, compute_phi, compute_pT


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

# Use multiprocessing with NUM_CPUS CPUs for both methods
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
                          title=r'Truth vs. Reconstructed Neutrino+ Pseudorapidity $\eta$',
                          xlim=(-3.5, 3.5))
plot_comparison_with_ratio(truth_eta_m, reco_eta_m, xlabel=r'Pseudorapidity $\eta$',
                          title=r'Truth vs. Reconstructed Neutrino- Pseudorapidity $\eta$',
                          xlim=(-3.5, 3.5))

# Plot phi for positive and negative neutrinos
plot_comparison_with_ratio(truth_phi_p, reco_phi_p, xlabel=r'Azimuthal Angle $\phi$ (radians)',
                          title=r'Truth vs. Reconstructed Neutrino+ $\phi$ (radians)')
plot_comparison_with_ratio(truth_phi_m, reco_phi_m, xlabel=r'Azimuthal Angle $\phi$ (radians)',
                          title=r'Truth vs. Reconstructed Neutrino- $\phi$ (radians)')

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



for i in range(n_events):
    # Get tau-tau system momentum for truth and reco
    p_tau_tau_truth = truth_data[i][0] + truth_data[i][1]
    p_tau_tau_reco = reco_tau_momenta[i][0] + reco_tau_momenta[i][1]
    

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


# Test collinearity assumption with truth data
truth_pion_p_momenta = [truth_data[i][2] for i in range(n_events)]
truth_pion_m_momenta = [truth_data[i][3] for i in range(n_events)]
truth_nu_p_momenta = [truth_data[i][4] for i in range(n_events)]
truth_nu_m_momenta = [truth_data[i][5] for i in range(n_events)]

# Plot relative uncertainties and residuals for neutrino components
for component, idx in [('px', 1), ('py', 2), ('pz', 3)]:
    # Neutrino+
    truth_p = [truth_data[i][4][idx] for i in range(n_events)]
    reco_p = [reco_neutrino_momenta[i][0][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_p, reco_p, component, 'Neutrino', '+')

    # Neutrino-
    truth_m = [truth_data[i][5][idx] for i in range(n_events)]
    reco_m = [reco_neutrino_momenta[i][1][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_m, reco_m, component, 'Neutrino', '-')

# Calculate cos theta distributions for tau+ and tau-
cos_theta_r_p_truth = []
cos_theta_n_p_truth = []
cos_theta_k_p_truth = []
cos_theta_r_p_reco = []
cos_theta_n_p_reco = []
cos_theta_k_p_reco = []

cos_theta_r_m_truth = []
cos_theta_n_m_truth = []
cos_theta_k_m_truth = []
cos_theta_r_m_reco = []
cos_theta_n_m_reco = []
cos_theta_k_m_reco = []

for i in range(n_events):
    # Tau+
    p_tau_p_truth = truth_data[i][0]  # Truth tau+ momentum
    p_pion_p_truth = truth_data[i][2]  # Truth pion+ momentum
    p_tau_p_reco = reco_tau_momenta[i][0]  # Reco tau+ momentum
    p_pion_p_reco = reco_data[i][0]  # Reco pion+ momentum
    
    # Define coordinate system for tau+
    r_hat_p, n_hat_p, k_hat_p = define_coordinate_system(p_tau_p_truth)
    
    # Boost pion+ momentum to tau+ rest frame (truth)
    p_pion_p_rest_truth = boost_to_rest_frame(p_pion_p_truth, p_tau_p_truth)
    # Boost pion+ momentum to tau+ rest frame (reco)
    p_pion_p_rest_reco = boost_to_rest_frame(p_pion_p_reco, p_tau_p_reco)
    
    # Compute cos theta for tau+ (truth)
    cos_r_p_truth, cos_n_p_truth, cos_k_p_truth = compute_cos_theta(p_pion_p_rest_truth, r_hat_p, n_hat_p, k_hat_p)
    cos_theta_r_p_truth.append(cos_r_p_truth)
    cos_theta_n_p_truth.append(cos_n_p_truth)
    cos_theta_k_p_truth.append(cos_k_p_truth)
    
    # Compute cos theta for tau+ (reco)
    cos_r_p_reco, cos_n_p_reco, cos_k_p_reco = compute_cos_theta(p_pion_p_rest_reco, r_hat_p, n_hat_p, k_hat_p)
    cos_theta_r_p_reco.append(cos_r_p_reco)
    cos_theta_n_p_reco.append(cos_n_p_reco)
    cos_theta_k_p_reco.append(cos_k_p_reco)
    
    # Tau-
    p_tau_m_truth = truth_data[i][1]  # Truth tau- momentum
    p_pion_m_truth = truth_data[i][3]  # Truth pion- momentum
    p_tau_m_reco = reco_tau_momenta[i][1]  # Reco tau- momentum
    p_pion_m_reco = reco_data[i][1]  # Reco pion- momentum
    
    # Define coordinate system for tau-
    r_hat_m, n_hat_m, k_hat_m = define_coordinate_system(p_tau_m_truth)
    
    # Boost pion- momentum to tau- rest frame (truth)
    p_pion_m_rest_truth = boost_to_rest_frame(p_pion_m_truth, p_tau_m_truth)
    # Boost pion- momentum to tau- rest frame (reco)
    p_pion_m_rest_reco = boost_to_rest_frame(p_pion_m_reco, p_tau_m_reco)
    
    # Compute cos theta for tau- (truth)
    cos_r_m_truth, cos_n_m_truth, cos_k_m_truth = compute_cos_theta(p_pion_m_rest_truth, r_hat_m, n_hat_m, k_hat_m)
    cos_theta_r_m_truth.append(cos_r_m_truth)
    cos_theta_n_m_truth.append(cos_n_m_truth)
    cos_theta_k_m_truth.append(cos_k_m_truth)
    
    # Compute cos theta for tau- (reco)
    cos_r_m_reco, cos_n_m_reco, cos_k_m_reco = compute_cos_theta(p_pion_m_rest_reco, r_hat_m, n_hat_m, k_hat_m)
    cos_theta_r_m_reco.append(cos_r_m_reco)
    cos_theta_n_m_reco.append(cos_n_m_reco)
    cos_theta_k_m_reco.append(cos_k_m_reco)

# Plot cos theta distributions for tau+
plot_comparison_with_ratio(cos_theta_r_p_truth, cos_theta_r_p_reco, xlabel=r'$\cos\theta_r$',
                          title=r'$\cos\theta_r$ Distribution for Tau+', bins=50, xlim=(-1, 1))
plot_comparison_with_ratio(cos_theta_n_p_truth, cos_theta_n_p_reco, xlabel=r'$\cos\theta_n$',
                          title=r'$\cos\theta_n$ Distribution for Tau+', bins=50, xlim=(-1, 1))
plot_comparison_with_ratio(cos_theta_k_p_truth, cos_theta_k_p_reco, xlabel=r'$\cos\theta_k$',
                          title=r'$\cos\theta_k$ Distribution for Tau+', bins=50, xlim=(-1, 1))

# Plot cos theta distributions for tau-
plot_comparison_with_ratio(cos_theta_r_m_truth, cos_theta_r_m_reco, xlabel=r'$\cos\theta_r$',
                          title=r'$\cos\theta_r$ Distribution for Tau-', bins=50, xlim=(-1, 1))
plot_comparison_with_ratio(cos_theta_n_m_truth, cos_theta_n_m_reco, xlabel=r'$\cos\theta_n$',
                          title=r'$\cos\theta_n$ Distribution for Tau-', bins=50, xlim=(-1, 1))
plot_comparison_with_ratio(cos_theta_k_m_truth, cos_theta_k_m_reco, xlabel=r'$\cos\theta_k$',
                          title=r'$\cos\theta_k$ Distribution for Tau-', bins=50, xlim=(-1, 1))

# Plot 2D correlation plots for cos theta observables
def plot_2d_correlation(truth_values, reco_values, xlabel, ylabel, title, bins=50, range=(-1, 1)):
    """Plot 2D correlation heatmap with y=x line"""
    plt.figure(figsize=(8, 8))
    plt.hist2d(truth_values, reco_values, bins=bins, range=[range, range], cmap='viridis', cmin=1)
    plt.colorbar(label='Counts')
    plt.plot(range, range, 'r--', linewidth=1, label='y = x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Create a safe filename from the title
    filename = title.lower().replace(' ', '_').replace('$', '').replace('/', '_') + '_2d.png'
    plt.savefig(f'plots/{filename}')
    plt.close()

# Plot correlations for each cos theta component
for component, label in [('r', r'$\cos\theta_r$'), ('n', r'$\cos\theta_n$'), ('k', r'$\cos\theta_k$')]:
    # Tau+
    plot_2d_correlation(
        cos_theta_r_p_truth, cos_theta_r_p_reco,
        xlabel=f'Truth {label}',
        ylabel=f'Reconstructed {label}',
        title=f'Truth vs Reconstructed {label} for Tau+'
    )
    # Tau-
    plot_2d_correlation(
        cos_theta_r_m_truth, cos_theta_r_m_reco,
        xlabel=f'Truth {label}',
        ylabel=f'Reconstructed {label}',
        title=f'Truth vs Reconstructed {label} for Tau-'
    )

# Calculate spin correlation matrix C_ij
def calculate_spin_correlation(cos_theta_r_p, cos_theta_n_p, cos_theta_k_p,
                              cos_theta_r_m, cos_theta_n_m, cos_theta_k_m):
    # Initialize correlation matrix
    C = np.zeros((3, 3))
    
    # Axes labels
    axes_p = [cos_theta_r_p, cos_theta_n_p, cos_theta_k_p]
    axes_m = [cos_theta_r_m, cos_theta_n_m, cos_theta_k_m]
    axis_labels = ['r', 'n', 'k']
    
    # Calculate C_ij for each pair of axes
    for i in range(3):
        for j in range(3):
            # Calculate x = cos_theta_i * cos_theta_j for each event
            x = [axes_p[i][k] * axes_m[j][k] for k in range(len(axes_p[i]))]
            
            # Count positive and negative x values
            N_plus = sum(1 for val in x if val > 0)
            N_minus = sum(1 for val in x if val < 0)
            
            # Calculate asymmetry A
            if (N_plus + N_minus) > 0:
                A = (N_plus - N_minus) / (N_plus + N_minus)
            else:
                A = 0
                
            # Calculate correlation matrix element
            C[i,j] = 4 * A
            
    return C, axis_labels

# Calculate and print correlation matrix for truth and reconstructed values
C_truth, labels = calculate_spin_correlation(cos_theta_r_p_truth, cos_theta_n_p_truth, cos_theta_k_p_truth,
                                            cos_theta_r_m_truth, cos_theta_n_m_truth, cos_theta_k_m_truth)
print("Truth Spin Correlation Matrix:")
print("   " + "   ".join(labels))
for i in range(3):
    print(f"{labels[i]} {C_truth[i]}")

C_reco, labels = calculate_spin_correlation(cos_theta_r_p_reco, cos_theta_n_p_reco, cos_theta_k_p_reco,
                                           cos_theta_r_m_reco, cos_theta_n_m_reco, cos_theta_k_m_reco)
print("\nReconstructed Spin Correlation Matrix:")
print("   " + "   ".join(labels))
for i in range(3):
    print(f"{labels[i]} {C_reco[i]}")

# Plot correlation matrices
def plot_correlation_matrix(C, labels, title):
    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add values to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{C[i,j]:.2f}",
                         ha="center", va="center", color="w")
    
    ax.set_title(title)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'plots/{title.lower().replace(" ", "_")}.png')
    plt.close()

plot_correlation_matrix(C_truth, labels, "Truth Spin Correlation Matrix")
plot_correlation_matrix(C_reco, labels, "Reconstructed Spin Correlation Matrix")

