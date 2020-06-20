import numpy as np
import networkx as nx
from scipy.special import comb
from collections import Counter
import community
import sklearn.preprocessing


def local_random_walk(network, n1, n2, p_tran):
    
    # Compute and return local random walk similarity score.
    pi_x = np.zeros(p_tran.shape[0], dtype=float)
    pi_x[n1] = 1.0
    pi_y = np.zeros(p_tran.shape[0], dtype=float)
    pi_y[n2] = 1.0
    for t in range(10):
        pi_x = p_tran.dot(pi_x)
        pi_y = p_tran.dot(pi_y)

    return network.degree[n1]/(2*network.number_of_edges()) + pi_x[n2] + network.degree[n2]/(2*network.number_of_edges()) + pi_y[n1]


def get_feature_extractor(network, features):
    """
    Get function that extracts specified features from a pair of nodes.

    Args:
        network (object): Networkx representation of the network.
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
        if feature == 'common-neighbors':

            # Return number of common neighbors.
            return len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))

        elif feature == 'jaccard-coefficient':
            
            # Return Jaccard coefficient for the node pair.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            size_un = len(set(network.neighbors(n1)).union(network.neighbors(n2)))
            return size_int/size_un if size_un > 0.0 else 0.0

        elif feature == 'hub-promoted':
        
            # Return Hub-promoted index.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            denom = min(len(set(network.neighbors(n1))), len(set(network.neighbors(n1))))
            if denom > 0:
                return size_int/denom
            else:
                return 0

        elif feature == 'adamic-adar':

            # Compute and return Adamic-Adar index.
            return np.sum([1/np.log(len(set(network.neighbors(n)))) 
                for n in set(network.neighbors(n1)).intersection(network.neighbors(n2))
                if len(set(network.neighbors(n))) > 1])

        elif feature == 'resource-allocation':
            
            # Compute and return resource-allocation index.
            return np.sum([1/len(set(network.neighbors(n))) 
                for n in set(network.neighbors(n1)).intersection(network.neighbors(n2))
                if len(set(network.neighbors(n))) > 0])
        
        elif feature == 'sorenson':
            
            # Compute and return Sorenson index.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            denom = len(set(network.neighbors(n1))) + len(set(network.neighbors(n1)))
            return size_int/denom if denom > 0.0 else 0.0
        
        elif feature == 'hub-depressed':
        
            # Return Hub-depressed index.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            denom = max(len(set(network.neighbors(n1))), len(set(network.neighbors(n1))))
            if denom > 0:
                return size_int/denom
            else:
                return 0
        
        elif feature == 'salton':
            
            # Compute and return Salton index.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            denom = np.sqrt(len(set(network.neighbors(n1))) * len(set(network.neighbors(n1))))
            return size_int/denom if denom > 0.0 else 0.0
        
        elif feature == 'leicht-holme-nerman':
            
            # Compute and return Leicht-Holme-Nerman index.
            size_int = len(set(network.neighbors(n1)).intersection(network.neighbors(n2)))
            denom = len(set(network.neighbors(n1))) * len(set(network.neighbors(n1)))
            return size_int/denom if denom > 0.0 else 0.0
        
        elif feature == 'preferential-attachment':
            
            # Compute and return preferential-attachment index.
            return len(set(network.neighbors(n1)))*len(set(network.neighbors(n2)))
        
        elif feature == 'local-random-walk':

            # Compute Local random walk score.
            return local_random_walk(network, n1, n2, p_tran)
        
        elif feature == 'superposed-random-walk':

            # Compute Local random walk score.
            return sum([local_random_walk(network, n1, n2, p_tran) for _ in range(5)])
        
        elif feature == 'simrank':

            # Return Simrank score.
            return simrank_scores[n1][n2]

        elif feature == 'same-community':

            # Return flag specifying whether the two nodes are part of
            # the same community or not.
            return int(communities[n1] == communities[n2])

        elif feature == 'community-index':

            # If nodes not part of same community, return 0.
            if communities[n1] != communities[n2]:
                return 0
            else:
                
                # Get community index of both nodes.
                communitiy_idx = communities[n1]

                # Compute community index.
                return m_counts[communitiy_idx]/comb(n_counts[communitiy_idx], 2)

        elif feature == 'page-rank':

            # Compare PageRank scores of the nodes.
            return abs(page_rank[n1] - page_rank[n2])

        elif feature == 'node2vec':
            return np.hstack((n2v_model.wv[str(n1)], n2v_model.wv[str(n1)]))

        elif feature == 'random':

            # Return random value as feature.
            return np.random.rand()
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
   
    ### PRECOMPUTED DATA FOR WHOLE NETWORK (NEEDED FOR SOME MEASURES) ###
    if 'simrank' in features:

        # Compute simrank scores.
        simrank_scores = nx.algorithms.similarity.simrank_similarity(network)

    if 'local-random-walk' in features or 'superposed-random-walk' in features:

        # Get adjacency matrix and compute probabilities of transitions.
        adj = nx.to_scipy_sparse_matrix(network)
        p_tran = sklearn.preprocessing.normalize(adj, norm='l1', axis=0)
    
    if 'same-community' in features or 'community-index' in features:

        # Get communities.
        communities = community.best_partition(network, randomize=True)
        
        # Initialize dictionary mapping community indices to counts of links contained within them.
        m_counts = dict.fromkeys(set(communities.values()), 0)

        # Count number of nodes in each community.
        n_counts = Counter(communities.values())

        # Go over links in network.
        for edge in network.edges():

            # If link within community, add to accumulator for that community.
            if communities[edge[0]] == communities[edge[1]]:
                m_counts[communities[edge[0]]] += 1

    if 'page-rank' in features:

        # Compute PageRank of nodes
        page_rank = nx.pagerank(network)

    if 'node2vec' in features:
        import node2vec
        n2v = node2vec.Node2Vec(network, dimensions=64, walk_length=30, num_walks=20, workers=8)
        n2v_model = n2v.fit(window=10, min_count=1, batch_words=4)



    #####################################################################
   
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

