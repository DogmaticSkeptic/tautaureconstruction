import numpy as np
import pickle
from helperfunctions import chi_squared_nu, compute_cos_theta, boost_to_rest_frame, define_coordinate_system, reconstruct_neutrino_momenta
from scipy.optimize import minimize
from tqdm import tqdm

def calculate_cos_theta_overlap(truth_cos, reco_cos):
    """Calculate overlap between truth and reconstructed cos theta distributions"""
    hist_truth, _ = np.histogram(truth_cos, bins=50, range=(-1, 1))
    hist_reco, _ = np.histogram(reco_cos, bins=50, range=(-1, 1))
    overlap = np.sum(np.minimum(hist_truth, hist_reco)) / np.sum(hist_truth)
    return overlap

def evaluate_uncertainties(sigma_tau, sigma_met, sigma_z, data):
    """Evaluate reconstruction quality for given uncertainties"""
    # Modify uncertainties in helperfunctions
    global SIGMA_TAU, SIGMA_MET, SIGMA_Z
    SIGMA_TAU = sigma_tau
    SIGMA_MET = sigma_met
    SIGMA_Z = sigma_z
    
    # Reconstruct neutrino momenta
    reco_neutrino_momenta = []
    for i in range(data['n_events']):
        result = reconstruct_neutrino_momenta(
            data['reco_data'][i][0], data['reco_data'][i][1], 
            data['MET'][i].px, data['MET'][i].py
        )
        reco_neutrino_momenta.append(result)
    
    # Calculate cos theta values
    truth_cos_theta_k_p, reco_cos_theta_k_p = [], []
    for i in range(data['n_events']):
        # Truth calculation
        p_tau_tau_truth = data['truth_data'][i][0] + data['truth_data'][i][1]
        p_tau_m_truth_rest = boost_to_rest_frame(data['truth_data'][i][1], p_tau_tau_truth)
        p_pion_m_truth_rest = boost_to_rest_frame(data['truth_data'][i][3], p_tau_tau_truth)
        p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_truth_rest, p_tau_m_truth_rest)
        r_hat, n_hat, k_hat = define_coordinate_system(p_tau_m_truth_rest)
        truth_cos_theta_k_p.append(compute_cos_theta(p_pion_m_single_rest, r_hat, n_hat, k_hat)[2])
        
        # Reco calculation
        p_tau_p_reco = data['reco_data'][i][0] + reco_neutrino_momenta[i][0]
        p_tau_m_reco = data['reco_data'][i][1] + reco_neutrino_momenta[i][1]
        p_tau_tau_reco = p_tau_p_reco + p_tau_m_reco
        p_tau_m_reco_rest = boost_to_rest_frame(p_tau_m_reco, p_tau_tau_reco)
        p_pion_m_reco_rest = boost_to_rest_frame(data['reco_data'][i][1], p_tau_tau_reco)
        p_pion_m_single_rest = boost_to_rest_frame(p_pion_m_reco_rest, p_tau_m_reco_rest)
        r_hat, n_hat, k_hat = define_coordinate_system(p_tau_m_reco_rest)
        reco_cos_theta_k_p.append(compute_cos_theta(p_pion_m_single_rest, r_hat, n_hat, k_hat)[2])
    
    return calculate_cos_theta_overlap(truth_cos_theta_k_p, reco_cos_theta_k_p)

def grid_search():
    """Perform grid search over uncertainties"""
    # Load data
    data = pickle.load(open('analysis_data.pkl', 'rb'))
    
    # Define search space
    tau_uncertainties = np.linspace(0.001, 0.01, 5)
    met_uncertainties = np.linspace(0.001, 0.01, 5)
    z_uncertainties = np.linspace(0.001, 0.01, 5)
    
    best_score = 0
    best_params = (0.006, 0.08, 0.001)  # Default values
    
    # Perform grid search
    for sigma_tau in tqdm(tau_uncertainties, desc="Tau uncertainty"):
        for sigma_met in tqdm(met_uncertainties, desc="MET uncertainty", leave=False):
            for sigma_z in tqdm(z_uncertainties, desc="Z uncertainty", leave=False):
                score = evaluate_uncertainties(sigma_tau, sigma_met, sigma_z, data)
                if score > best_score:
                    best_score = score
                    best_params = (sigma_tau, sigma_met, sigma_z)
    
    return best_params, best_score

if __name__ == "__main__":
    best_params, best_score = grid_search()
    print(f"Best uncertainties: Tau={best_params[0]:.4f}, MET={best_params[1]:.4f}, Z={best_params[2]:.4f}")
    print(f"Best cos theta overlap: {best_score:.4f}")
