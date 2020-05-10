import numpy as np


def get_feature_extractor(features):
    """
    Get function that extracts specified features from a pair of nodes.

    Args:
        features (list): List of names of features to extract.
    
    Returns:
        (function): Function that takes a network and two nodes (node pair) and
        computes the specified features in the form of a numpy array.
    """

    def get_feature(network, n1, n2, feature):
        """
        Get specified feature for pair of nodes n1 and n2.
        This function is used by the get_feature_extractor function.

        Args:
            network (object): Networkx representation of the network.
            n1 (str): First node in pair.
            n2 (str): Second node in pair.
            feature (str): Name of feature to extract.

        Returns:
            (float): The extracted feature.
        """
        
        # Extract specified feature.
        if feature == 'feature1':
            return 1.0
        elif feature == 'feature2':
            return 2.0
        elif feature == 'feature3':
            return 3.0
        else:
            raise ValueError('Unknown feature ' + feature)
    
    def feature_extractor(network, n1, n2, features):
        """
        The feature extractor function. This function is partially applied
        with the list of features and returned by the get_feature_extractor function.

        Args:
            network (object): Networkx representation of the network.
            n1 (str): First node in pair.
            n2 (str): Second node in pair.
            features (list): List of names of features to extract.
        """

        return np.hstack([get_feature(network, n1, n2, feature) for feature in features])
   
    return (lambda network, n1, n2 : feature_extractor(network, n1, n2, features))
 

def get_features(network, edges, feature_extractor_func):
    """
    Extract features for specified edges. The function takes a network, a list
    of edges as tuples and a feature extraction function. The function uses the feature
    extraction function feature_extractor_func to compute the features for node pairs
    connected by each edge or 'non-edge'. If the node pair is connected in the network, the edge
    is removed prior to the application of the feature extraction function. The features
    for the node pairs are returned in the form of a numpy array where each row corresponds
    to a node pair in the same order as specified in the edges parameter.

    Args:
        network (object): Networkx representation of the network containing the node pairs.
        edges (list): List of node pairs representing the edges or 'non-edges' for which 
        to compute the features.
        feature_extractor_func (function): Function that takes a network and two nodes and
        returns a numpy array containing the features describing the pair.

    Returns:
        (numpy.ndarray): Numpy array where each row corresponds to a node pair in 
        the same order as specified in the edges parameter.
    """

    # Initialize array for storing computed feature vectos.
    feature_vectors = []

    # Go over specified edges.
    for edge in edges:

        # If edge in network, remove, construct features
        # and add back. If not, just compute features.
        edge_removed = False
        if network.has_edge(*edge):
            network.remove_edge(*edge)
            edge_removed = True
        feature_vectors.append(feature_extractor_func(network, *edge))
        if edge_removed:
            network.add_edge(*edge)
    
    # Stack computed feature vectors as 2d numpy array (matrix) and return.
    return np.vstack(feature_vectors)

