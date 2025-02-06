import numpy as np

def boost_to_rest_frame(p, p_boost, debug=False):
    """Boost a 4-momentum p into the rest frame of p_boost"""
    # Check for invalid inputs
    if np.any(np.isnan(p)) or np.any(np.isnan(p_boost)):
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    # Early-out for trivial boost
    if np.linalg.norm(p_boost[1:]) < 1e-10:
        return p  # Already in the rest frame
    
    # Calculate boost vector (direction and magnitude)
    beta = p_boost[1:] / p_boost[0]  # β = p/E
    beta_sq = np.dot(beta, beta)
    
    # Check for invalid beta_sq
    if beta_sq >= 1.0 or beta_sq < 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    
    # Calculate components parallel and perpendicular to boost
    beta_dot_p = np.dot(beta, p[1:])
    p_parallel = (beta_dot_p / beta_sq) * beta
    p_perp = p[1:] - p_parallel
    
    # Apply Lorentz transformation
    E_prime = gamma * (p[0] - beta_dot_p)
    p_prime_parallel = gamma * (p_parallel - beta * p[0])
    p_prime = np.array([E_prime, *(p_prime_parallel + p_perp)])
    
    return p_prime

def boost_three_vector(vec3, p_boost):
    """Boost a 3-vector by treating it as a 4-vector with E=0"""
    # Check for invalid inputs
    if np.any(np.isnan(vec3)) or np.any(np.isnan(p_boost)):
        return np.array([np.nan, np.nan, np.nan])
    
    # Calculate boost parameters
    beta = p_boost[1:] / p_boost[0]
    beta_sq = np.dot(beta, beta)
    
    if beta_sq >= 1.0:  # also covers beta_sq < 0 implicitly
        return np.array([np.nan, np.nan, np.nan])
    
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    
    # Form the 4-vector with E=0:
    V = np.concatenate(([0.0], vec3))
    
    # Boost the 4-vector: note the standard formulas
    V0_prime = -gamma * np.dot(beta, vec3)
    V_space_prime = vec3 + ((gamma - 1.0) * np.dot(beta, vec3) / beta_sq) * beta
    
    # Now extract and renormalize the spatial part:
    v_prime = V_space_prime
    norm = np.linalg.norm(v_prime)
    if norm > 0:
        return v_prime / norm
    else:
        return np.array([np.nan, np.nan, np.nan])
