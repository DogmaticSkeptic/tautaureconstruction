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
    """Plot 3D scatter plot with sphere sizes representing scores"""
    # Convert results to numpy array
    grid = np.array([r[0] for r in results])
    scores = np.array([r[1] for r in results])
    
    # Normalize scores for sphere sizes (inverse since smaller is better)
    max_score = scores.max()
    sphere_sizes = 1000 * (1 - scores/max_score) + 10  # Scale sizes for visibility
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with spheres
    scatter = ax.scatter(
        grid[:, 0],  # SIGMA_TAU
        grid[:, 1],  # SIGMA_Z
        grid[:, 2],  # SIGMA_MET
        s=sphere_sizes,
        c=scores,
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Total Score')
    
    # Labels
    ax.set_xlabel('SIGMA_TAU')
    ax.set_ylabel('SIGMA_Z')
    ax.set_zlabel('SIGMA_MET')
    ax.set_title('3D Grid Search Results\n(Sphere size indicates score quality)')
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('grid_search_3d.png')
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
