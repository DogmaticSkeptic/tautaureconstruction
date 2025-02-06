import numpy as np

def compute_cos_theta(p_pion, r_hat, n_hat, k_hat):
    """Calculate cos theta for each axis in the rest frame"""
    # Check for invalid inputs
    if np.any(np.isnan(p_pion)) or np.any(np.isnan(r_hat)) or \
       np.any(np.isnan(n_hat)) or np.any(np.isnan(k_hat)):
        return np.nan, np.nan, np.nan
    
    # Calculate pion momentum norm
    p_norm = np.linalg.norm(p_pion[1:])
    if p_norm < 1e-10:  # Avoid division by zero
        return np.nan, np.nan, np.nan
    
    # Normalize the pion momentum vector
    p_pion_norm = p_pion[1:] / p_norm
    
    # Calculate cos theta for each axis
    cos_theta_r = np.dot(p_pion_norm, r_hat)
    cos_theta_n = np.dot(p_pion_norm, n_hat)
    cos_theta_k = np.dot(p_pion_norm, k_hat)
    
    # Ensure cosines are in valid range
    cos_theta_r = np.clip(cos_theta_r, -1.0, 1.0)
    cos_theta_n = np.clip(cos_theta_n, -1.0, 1.0)
    cos_theta_k = np.clip(cos_theta_k, -1.0, 1.0)
    
    return cos_theta_r, cos_theta_n, cos_theta_k
