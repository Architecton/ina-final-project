import networkx as nx

import random
import math
import itertools


def train_test_split(network, test_size=0.1):
    """
    Construct test network by removing specified fraction of edges.
    Return network with removed edges as well as edges that were removed
    as well as an equal number of node pairs (edges) that do not exist
    in the original network.

    Args:
        network (object): Networkx representation of the network.
        test_size (float): Fraction of edges removed for testing.

    Returns:
        (tuple): training and test sets, test network.

    Raises:
        ValueError: This exception is raised if the size of the test set is
        not valid (not from interval (0.0, 1.0)).
    """

    # Check arguments.
    if test_size <= 0 or test_size >= 1.0:
        raise ValueError('The test_size parameter should be larger than 0.0 and smaller than 1.0')
    
    # Get list of nodes in network.
    network_nodes = list(network.nodes())

    # Get test 'non-edges'.
    test_edges_neg = set()
    while len(test_edges_neg) < math.ceil(test_size*network.number_of_edges()):
        
        # Sample node, get non-neighbors and sample.
        first = random.choice(network_nodes)
        excluded = list(network.neighbors(first)) + [first]
        valid = list(set(network_nodes).difference(excluded))
        second = random.choice(valid)
        test_edges_neg.add(frozenset((first, second)))
    
    # Sample edges and remove them from network to get test network.
    test_edges_pos = random.sample(network.edges(), math.ceil(test_size*network.number_of_edges()))
    
    # Get test network and training edges.
    test_network = network.copy()
    test_network.remove_edges_from(test_edges_pos)
    train_edges_pos = list(test_network.edges())
    
    # Get train 'non-edges'.
    train_edges_neg = set()
    while len(train_edges_neg) < len(train_edges_pos):
        
        # Sample node, get non-neighbors and sample.
        first = random.choice(network_nodes)
        excluded = list(network.neighbors(first)) + [first]
        valid = list(set(network_nodes).difference(excluded))
        second = random.choice(valid)
        pair = {first, second}
        if pair not in test_edges_neg:
            train_edges_neg.add(frozenset(pair))

    # Return training and test set indices and test network.
    return train_edges_pos, test_edges_pos, list(map(tuple, train_edges_neg)), list(map(tuple, test_edges_neg)), test_network


def repeated_train_test_split(network, n_rep, test_size=0.1):
    """
    Perform repetitions of train_test splits on the specified network.
    Repeatedly construct test network by removing specified fraction of edges.
    Return network with removed edges as well as edges that were removed
    as well as an equal number of node pairs (edges) that do not exist
    in the original network.

    Args:
        network (object): Networkx representation of the network.
        n_rep (int): Number of repetitions of the train-test split.
        test_size (float): Fraction of edges removed for testing.

    yields:
        (tuple): training and test sets, test network.
    
    Raises:
        ValueError: This exception is raised if specified number of repetitions
        is invalid (negative).
    """
    
    # Check arguments.
    if n_rep <= 0:
        raise ValueError('Number of repetitions should be greater than 0')
    
    # Repeat train_test split specified number of times.
    for _ in range(n_rep):
        yield train_test_split(network, test_size)

