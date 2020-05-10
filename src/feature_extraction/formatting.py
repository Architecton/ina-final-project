import numpy as np
import sklearn.utils

def format_data(data_pos, data_neg, shuffle=False):
    """
    Concatenate positive samples and negative samples into
    cannonical matrix data representation and construct
    array of target variable values. Shuffle if specified.

    Args:
       data_pos (numpy.ndarray): Array of features corresponding to
       linked node pairs.
       data_neg (numpy.ndarray): Array of features corresponding to
       node pairs without links.
       shuffle (bool): Shuffle the data or not
    
    Returns:
        (tuple): Canonical representation of the data as a matrix
        of features and an array of target variable values.
    """

    # Set target variable values.
    target_pos = np.ones(data_pos.shape[0], dtype=int)
    target_neg = np.zeros(data_neg.shape[0], dtype=int)
    
    # Stack all training samples and target values.
    data = np.vstack((data_pos, data_neg))
    target = np.hstack((target_pos, target_neg))
    
    # Shuffle if specified.
    if shuffle:
        data, target = sklearn.utils.shuffle(data, target)
    
    # Return cannonical representation of training data and target
    # variable values.
    return data, target
    
