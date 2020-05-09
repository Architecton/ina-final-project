import random

def train_test_split(network, test_size):
    """
    Construct test network by removing specified fraction of edges.
    Return network with removed edges as well as edges that were removed
    as well as an equal number of node pairs (edges) that do not exist
    in the original network.

    Args:
        network (object): Networkx representation of the network.
        test_size (float): Fraction of edges removed for testing.

    Returns:
        (tuple): augmented network, removed edges and equal number of node pairs that
        were not linked in the original network.
    """
   
    # Sample specified number of "non-edges".
    edges_neg = []
    while len(edges_neg) < math.ceil(test_size*network.number_of_edges()):
        pair = tuple(random.sample(network.nodes, 2))
        if not network.has_edge(*pair):
            edges_neg.append(pair)

    # Sample edges and remove them from network to get test network.
    edges_pos = random.sample(network.edges, math.ceil(test_size*network.number_of_edges()))
    network_test = network.copy().remove_edges_from(edges_pos)

    # Return test network, removed edges and sampled "non-edges".
    return test_network, edges_pos, edges_neg

if __name__ == '__main__':
    # TODO
    pass
