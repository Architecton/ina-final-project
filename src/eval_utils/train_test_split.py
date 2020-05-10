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
    """

    
    # Check arguments.
    if test_size <= 0 or test_size >= 1.0:
        raise ValueError('The test_size parameter should be larger than 0.0 and smaller than 1.0')
   
    # Sample specified number of "non-edges".
    test_edges_neg = []
    while len(test_edges_neg) < math.ceil(test_size*network.number_of_edges()):
        pair = tuple(random.sample(network.nodes, 2))
        if not network.has_edge(*pair):
            test_edges_neg.append(pair)
    
    # Get all other "non-edges".
    train_edges_neg_all = []
    test_edges_neg_comp = map(frozenset, test_edges_neg)
    for pair in itertools.combinations(network.nodes(), r=2):
        if not network.has_edge(*pair) and frozenset(pair) not in test_edges_neg_comp:
            train_edges_neg_all.append(pair)
    

    # Sample edges and remove them from network to get test network.
    test_edges_pos = random.sample(network.edges, math.ceil(test_size*network.number_of_edges()))
    
    # Get test network and training edges.
    test_network = network.copy()
    test_network.remove_edges_from(test_edges_pos)
    train_edges_pos = list(test_network.edges())

    # Sample training "non-edges" from "non-edges" not in test-set to match number
    # of training edges.
    train_edges_neg = random.sample(train_edges_neg_all, len(train_edges_pos))
    
    # Return training and test set indices and test network.
    return train_edges_pos, test_edges_pos, train_edges_neg, test_edges_neg, test_network


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
    """
    
    # Check arguments.
    if n_rep <= 0:
        raise ValueError('Number of repetitions should be greater than 0')
    
    # Repeat train_test split specified number of times.
    for _ in range(n_rep):
        yield train_test_split(network, test_size)


