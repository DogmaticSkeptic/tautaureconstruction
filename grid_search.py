import numpy as np
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import itertools
import helperfunctions as hf

NUM_CPUS = 12
GRID_POINTS = 10
N_EVENTS = 1000  # Use first 1000 events

def generate_grid():
    """Generate 3D grid of sigma values"""
    return list(itertools.product(
        np.linspace(0.001, 0.05, GRID_POINTS),  # SIGMA_TAU
        np.linspace(0.001, 0.05, GRID_POINTS),  # SIGMA_Z
        np.linspace(0.001, 0.05, GRID_POINTS)   # SIGMA_MET
    ))

def load_data():
    """Load data identical to main.py"""
    particle_data_dict = pickle.load(open('pi_pi_recon_particles.pkl', 'rb'))
    
    # Create args list with truth and reco data
    args_list = []
    for i in tqdm(range(N_EVENTS)):
        # Reconstructed components
        p_pi_p_reco = hf.compute_four_momentum(particle_data_dict['tau_p_child1'][i])
        p_pi_m_reco = hf.compute_four_momentum(particle_data_dict['tau_m_child1'][i])
        MET_x = particle_data_dict['MET'][i].px
        MET_y = particle_data_dict['MET'][i].py
        
        # Truth values
        truth_nu_p = hf.compute_four_momentum(particle_data_dict['truth_nu_p'][i])
        truth_nu_m = hf.compute_four_momentum(particle_data_dict['truth_nu_m'][i])
        
        args_list.append((p_pi_p_reco, p_pi_m_reco, MET_x, MET_y, truth_nu_p, truth_nu_m))
    
    return args_list

def evaluate_grid_point(grid_point, args_list):
    """Evaluate one grid point's performance"""
    sigma_tau, sigma_z, sigma_met = grid_point
    total_score = 0.0
    
    for args in args_list:
        p_pi_p, p_pi_m, MET_x, MET_y, truth_nu_p, truth_nu_m = args
        
        # Reconstruct with current sigmas
        try:
            p_nu_p, p_nu_m = hf.reconstruct_neutrino_momenta(
                p_pi_p, p_pi_m, MET_x, MET_y,
                sigma_tau=sigma_tau,
                sigma_z=sigma_z,
                sigma_met=sigma_met
            )
        except:
            return grid_point, np.inf  # Penalize failed reconstructions

        # Calculate residuals for all components (px, py, pz)
        for reco_nu, truth_nu in [(p_nu_p, truth_nu_p), (p_nu_m, truth_nu_m)]:
            for i in [1, 2, 3]:  # px, py, pz components
                truth_val = truth_nu[i]
                reco_val = reco_nu[i]
                if abs(truth_val) > 1e-6:  # Avoid division by zero
                    residual = abs((truth_val - reco_val) / truth_val)
                    total_score += residual

    return grid_point, total_score

def parallel_worker(grid_point, args_list):
    try:
        return evaluate_grid_point(grid_point, args_list)
    except Exception as e:
        print(f"Error with {grid_point}: {str(e)}")
        return grid_point, np.inf

import matplotlib.pyplot as plt

def plot_grid_search_results(results, grid_points):
    """Plot 2D heatmaps of grid search results"""
    # Convert results to numpy array for easier manipulation
    grid = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    
    # Reshape into 3D grid
    grid_3d = grid.reshape((grid_points, grid_points, grid_points, 3))
    scores_3d = scores.reshape((grid_points, grid_points, grid_points))
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot heatmaps for each fixed parameter
    for i, (param_name, ax) in enumerate(zip(['SIGMA_TAU', 'SIGMA_Z', 'SIGMA_MET'], axes)):
        # Find index where other parameters are at their median value
        fixed_idx = grid_points // 2
        
        if i == 0:  # Fix SIGMA_TAU
            data = scores_3d[fixed_idx, :, :]
            x = grid_3d[fixed_idx, :, :, 1]  # SIGMA_Z
            y = grid_3d[fixed_idx, :, :, 2]  # SIGMA_MET
            title = f'Score vs SIGMA_Z and SIGMA_MET\n(SIGMA_TAU fixed at {grid_3d[fixed_idx,0,0,0]:.4f})'
            xlabel = 'SIGMA_Z'
            ylabel = 'SIGMA_MET'
        elif i == 1:  # Fix SIGMA_Z
            data = scores_3d[:, fixed_idx, :]
            x = grid_3d[:, fixed_idx, :, 0]  # SIGMA_TAU
            y = grid_3d[:, fixed_idx, :, 2]  # SIGMA_MET
            title = f'Score vs SIGMA_TAU and SIGMA_MET\n(SIGMA_Z fixed at {grid_3d[0,fixed_idx,0,1]:.4f})'
            xlabel = 'SIGMA_TAU'
            ylabel = 'SIGMA_MET'
        else:  # Fix SIGMA_MET
            data = scores_3d[:, :, fixed_idx]
            x = grid_3d[:, :, fixed_idx, 0]  # SIGMA_TAU
            y = grid_3d[:, :, fixed_idx, 1]  # SIGMA_Z
            title = f'Score vs SIGMA_TAU and SIGMA_Z\n(SIGMA_MET fixed at {grid_3d[0,0,fixed_idx,2]:.4f})'
            xlabel = 'SIGMA_TAU'
            ylabel = 'SIGMA_Z'
        
        # Plot heatmap
        im = ax.imshow(data, origin='lower', aspect='auto',
                      extent=[x.min(), x.max(), y.min(), y.max()])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    plt.savefig('grid_search_heatmaps.png')
    plt.close()

if __name__ == "__main__":
    args_list = load_data()
    grid = generate_grid()
    print(f"Loaded {len(args_list)} events, searching {len(grid)} parameter combinations")
    
    # Create worker with frozen args_list
    worker = partial(parallel_worker, args_list=args_list)
    
    results = []
    with Pool(processes=NUM_CPUS) as pool:
        for result in tqdm(pool.imap(worker, grid), total=len(grid)):
            results.append(result)
    
    # Find and display best parameters
    results.sort(key=lambda x: x[1])
    best_params, best_score = results[0]
    print(f"\nBest parameters (tau, z, met): {best_params}")
    print(f"Best score: {best_score:.2f}")
    
    # Save and plot results
    with open("grid_search_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved results to grid_search_results.pkl")
    
    # Generate visualization
    plot_grid_search_results(results, GRID_POINTS)
    print("Saved grid search visualization to grid_search_heatmaps.png")
