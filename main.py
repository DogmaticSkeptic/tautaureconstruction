import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import pickle
NUM_CPUS = 4
from helperfunctions import (
    boost_to_rest_frame, define_coordinate_system,
    reconstruct_neutrino_momenta, plot_comparison_with_ratio, plot_relative_uncertainty,
    chi_squared_nu, chi_squared_collinear, compute_four_momentum, compute_eta, compute_phi, compute_pT,
    reconstruct_neutrino_collinear, plot_residual_comparison, plot_collinearity_test
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

def reconstruct_collinear_event(args):
    p_pi_p_reco, p_pi_m_reco, MET_x, MET_y = args
    return reconstruct_neutrino_collinear(p_pi_p_reco, p_pi_m_reco, MET_x, MET_y)

# Use multiprocessing with NUM_CPUS CPUs for both methods
with Pool(processes=NUM_CPUS) as pool:
    reco_neutrino_momenta = list(tqdm(pool.imap(reconstruct_event, reco_args), total=n_events))
    reco_neutrino_collinear = list(tqdm(pool.imap(reconstruct_collinear_event, reco_args), total=n_events))

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

plot_collinearity_test(truth_pion_p_momenta, truth_nu_p_momenta, 'Neutrino', '+')
plot_collinearity_test(truth_pion_m_momenta, truth_nu_m_momenta, 'Neutrino', '-')

# Plot relative uncertainties and residuals for neutrino components
for component, idx in [('px', 1), ('py', 2), ('pz', 3)]:
    # Neutrino+
    truth_p = [truth_data[i][4][idx] for i in range(n_events)]
    reco_p = [reco_neutrino_momenta[i][0][idx] for i in range(n_events)]
    reco_coll_p = [reco_neutrino_collinear[i][0][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_p, reco_p, component, 'Neutrino', '+')
    
    # Plot residual comparison
    residuals_original = [t - r for t,r in zip(truth_p, reco_p)]
    residuals_coll = [t - r for t,r in zip(truth_p, reco_coll_p)]
    plot_residual_comparison(residuals_original, residuals_coll,
                           xlabel=f'{component} Relative Residual',
                           title=f'Neutrino+ {component} Relative Residual Comparison',
                           truth_values=truth_p,
                           xlim=(-1, 1))
    
    # Print chi2 values for all events
    print(f"\nNeutrino+ {component} Chi2 values:")
    print(f"{'Event':<6} {'Original':<10} {'Collinear':<10}")
    for i in range(n_events):
        # Original method
        p_nu_p = reco_neutrino_momenta[i][0][1:]  # px,py,pz of nu+
        p_nu_m = reco_neutrino_momenta[i][1][1:]  # px,py,pz of nu-
        params = np.concatenate([p_nu_p, p_nu_m])
        chi2_orig = chi_squared_nu(params, reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py)
        
        # Collinear method
        p_nu_p, p_nu_m = reco_neutrino_collinear[i]
        alpha = np.linalg.norm(p_nu_p[1:])/np.linalg.norm(reco_data[i][0][1:])
        beta = np.linalg.norm(p_nu_m[1:])/np.linalg.norm(reco_data[i][1][1:])
        chi2_coll = chi_squared_collinear([alpha, beta], reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py)
        
        print(f"{i:<6} {chi2_orig:<10.2f} {chi2_coll:<10.2f}")

    # Neutrino-
    truth_m = [truth_data[i][5][idx] for i in range(n_events)]
    reco_m = [reco_neutrino_momenta[i][1][idx] for i in range(n_events)]
    reco_coll_m = [reco_neutrino_collinear[i][1][idx] for i in range(n_events)]
    plot_relative_uncertainty(truth_m, reco_m, component, 'Neutrino', '-')
    
    # Plot residual comparison
    residuals_original = [t - r for t,r in zip(truth_m, reco_m)]
    residuals_coll = [t - r for t,r in zip(truth_m, reco_coll_m)]
    plot_residual_comparison(residuals_original, residuals_coll,
                           xlabel=f'{component} Relative Residual',
                           title=f'Neutrino- {component} Relative Residual Comparison',
                           truth_values=truth_m,
                           xlim=(-1, 1))
    
    # Print chi2 values for all events
    print(f"\nNeutrino- {component} Chi2 values:")
    print(f"{'Event':<6} {'Original':<10} {'Collinear':<10}")
    for i in range(n_events):
        # Original method
        p_nu_p = reco_neutrino_momenta[i][0][1:]  # px,py,pz of nu+
        p_nu_m = reco_neutrino_momenta[i][1][1:]  # px,py,pz of nu-
        params = np.concatenate([p_nu_p, p_nu_m])
        chi2_orig = chi_squared_nu(params, reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py)
        
        # Collinear method
        p_nu_p, p_nu_m = reco_neutrino_collinear[i]
        alpha = np.linalg.norm(p_nu_p[1:])/np.linalg.norm(reco_data[i][0][1:])
        beta = np.linalg.norm(p_nu_m[1:])/np.linalg.norm(reco_data[i][1][1:])
        chi2_coll = chi_squared_collinear([alpha, beta], reco_data[i][0], reco_data[i][1], MET[i].px, MET[i].py)
        
        print(f"{i:<6} {chi2_orig:<10.2f} {chi2_coll:<10.2f}")


