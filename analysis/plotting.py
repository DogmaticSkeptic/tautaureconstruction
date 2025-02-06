import numpy as np
import matplotlib.pyplot as plt

def plot_comparison_with_ratio(truth_values, reco_values, xlabel, title, bins=50, xlim=None):
    """Plot histograms with ratio plot below"""
    # Filter out NaN values
    truth_values = np.array(truth_values)
    reco_values = np.array(reco_values)
    valid_mask = ~np.isnan(truth_values) & ~np.isnan(reco_values)
    truth_values = truth_values[valid_mask]
    reco_values = reco_values[valid_mask]
    
    if len(truth_values) == 0 or len(reco_values) == 0:
        print(f"Warning: No valid data to plot for {title}")
        return
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Top Plot - Truth (filled) vs Reco (outline)
    hist_truth, bin_edges = np.histogram(truth_values, bins=bins)
    hist_reco, _ = np.histogram(reco_values, bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax[0].hist(truth_values, bins=bin_edges, alpha=0.5, color='gray', label='Truth')
    ax[0].step(bin_centers, hist_reco, where='mid', color='orange', linewidth=2, label='Reconstructed')

    ax[0].set_ylabel('Count')
    ax[0].set_title(title)
    ax[0].legend()

    # Bottom Plot - Ratio (Reco/Truth)
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
    plt.show()

def plot_relative_uncertainty(truth_values, reco_values, component, particle_type, charge, bins=50, xlim=(-1, 1)):
    """Plot relative uncertainties between truth and reconstructed values"""
    # Calculate relative uncertainties
    rel_unc = [(reco - truth)/truth if truth != 0 else 0 
               for truth, reco in zip(truth_values, reco_values)]
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rel_unc, bins=bins, alpha=0.7)
    plt.xlabel(f'Relative Uncertainty in {component}')
    plt.ylabel('Count')
    plt.title(f'Relative Uncertainty in {component} for {particle_type}{charge}')
    plt.xlim(xlim)
    plt.grid(True)
    plt.show()
