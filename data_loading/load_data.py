import pickle
import numpy as np

def load_particle_data(file_path):
    """Load particle data from pickle file"""
    return pickle.load(open(file_path, 'rb'))

def compute_four_momentum(vec):
    """Compute four-momentum from particle vector"""
    px, py, pz, m = vec.px, vec.py, vec.pz, vec.mass
    p_magnitude = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p_magnitude**2 + m**2)
    return np.array([E, px, py, pz])
