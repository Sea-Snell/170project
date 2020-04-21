import networkx as nx
from collections import defaultdict


def is_valid_network(G, T):
    """
    Checks whether T is a valid network of G.
    Args:
        G: networkx.Graph
        T: networkx.Graph

    Returns:
        bool: whether T is a valid network
    """

    return nx.is_tree(T) and nx.is_dominating_set(G, T.nodes)


def average_pairwise_distance(T):
    """
    Computes the average pairwise distance between vertices in T.
    This is what we want to minimize!

    Note that this function is a little naive, i.e. there are much
    faster ways to compute the average pairwise distance in a tree.
    Feel free to write your own!

    Args:
        T: networkx.Graph, a tree

    Returns:
        double: the average pairwise distance
    """
    path_lengths = nx.all_pairs_dijkstra_path_length(T)
    total_pairwise_distance = sum([sum(length[1].values()) for length in path_lengths])
    return total_pairwise_distance / (len(T) * (len(T) - 1))


def average_pairwise_distance_fast(T):
    """Calculates the average pairwise distance for a tree in linear time.

    Since there is always unique path between nodes in a tree, each edge in the
    tree is used in all of the paths from the connected component on one side
    of the tree to the other. So each edge contributes to the total pairwise cost
    in the following way: if the size of the connected components that are
    created from removing an edge e are A and B, then the total pairwise distance
    cost for an edge is 2 * A * B * w(e) = (# of paths that use that edge) * w(e).
    We multiply by two to consider both directions that paths can take on an
    undirected edge.

    Since each edge connects a subtree to the rest of a tree, we can run DFS
    to compute the sizes of all of the subtrees, and iterate through all the edges
    and sum the pairwise distance costs for each edge and divide by the total
    number of pairs.

    This is very similar to Q7 on MT1.

    h/t to Noah Kingdon for the algorithm.
    """
    if not nx.is_connected(T):
        raise ValueError("Tree must be connected")

    if len(T) == 1: return 0

    subtree_sizes = {}
    marked = defaultdict(bool)
    # store child parent relationships for each edge, because the components
    # created when removing an edge are the child subtree and the rest of the vertices
    root = list(T.nodes)[0];
    
    child_parent_pairs = [(root, root)]

    def calculate_subtree_sizes(u):
        """Iterates through the tree to compute all subtree sizes in linear time

        Args:
            u: the root of the subtree to start the DFS

        """
        unmarked_neighbors = filter(lambda v: not marked[v], T.neighbors(u))
        marked[u] = True
        size = 0
        for v in unmarked_neighbors:
            child_parent_pairs.append((v, u))
            calculate_subtree_sizes(v)
            size += subtree_sizes[v]
        subtree_sizes[u] = size + 1
        return subtree_sizes[u]

    calculate_subtree_sizes(root)  # any vertex can be the root of a tree

    cost = 0
    for c, p in child_parent_pairs:
        if c != p:
            a, b = subtree_sizes[c], len(T.nodes) - subtree_sizes[c]
            w = T[c][p]["weight"]
            cost += 2 * a * b * w
    return cost / (len(T) * (len(T) - 1))
