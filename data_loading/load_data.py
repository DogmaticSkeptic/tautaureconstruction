import pickle
import numpy as np

def load_particle_data(file_path):
    """Load particle data from pickle file"""
    return pickle.load(open(file_path, 'rb'))
