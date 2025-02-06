import numpy as np

def define_coordinate_system(p_tau_p, p_tau_m):
    """Define the {r^, n^, k^} coordinate system in the tau tau rest frame"""
    # k^ is the flight direction of tau- in the tau tau rest frame
    k_hat = p_tau_m[1:] / np.linalg.norm(p_tau_m[1:])
    
    # p^ is the direction of one of the e± beams (assume z-axis)
    p_hat = np.array([0, 0, 1])
    
    # r^ = (p^ - k^ * cosΘ) / sinΘ
    cos_theta = np.dot(k_hat, p_hat)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Handle case where sinΘ is very small (tau nearly parallel to beam)
    if sin_theta < 1e-6:
        # Use x-axis as reference direction instead of z-axis
        p_hat = np.array([1, 0, 0])
        cos_theta = np.dot(k_hat, p_hat)
        sin_theta = np.sqrt(1 - cos_theta**2)
    
    r_hat = (p_hat - k_hat * cos_theta) / sin_theta
    
    # n^ = k^ × r^
    n_hat = np.cross(k_hat, r_hat)
    
    return r_hat, n_hat, k_hat
