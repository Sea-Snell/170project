import re
import os

import networkx as nx

import utils


def validate_file(path):
    """File must not exceed 100KB and must contain only numbers and spaces"""
    if os.path.getsize(path) > 100000:
        print(f"{path} exceeds 100KB, make sure you're not repeating edges!")
        return False
    with open(path, "r") as f:
        if not re.match(r"^[\d\.\s]+$", f.read()):
            print(f"{path} contains characters that are not numbers and spaces")
            return False
    return True


def read_input_file(path, max_size=None):
    """
    Parses and validates an input file

    :param path: str, a path
    :return: networkx Graph is the input is well formed, AssertionError thrown otherwise
    """
    with open(path, "r") as fo:
        n = fo.readline().strip()
        assert n.isdigit()
        n = int(n)

        lines = fo.read().splitlines()
        fo.close()

        # validate lines
        for line in lines:
            tokens = line.split(" ")

            assert len(tokens) == 3
            assert tokens[0].isdigit() and int(tokens[0]) < n
            assert tokens[1].isdigit() and int(tokens[1]) < n
            assert bool(re.match(r"(^\d+\.\d{1,3}$|^\d+$)", tokens[2]))
            assert 0 < float(tokens[2]) < 100

        G = nx.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
        G.add_nodes_from(range(n))

        assert nx.is_connected(G)

        if max_size is not None:
            assert len(G) <= max_size

        return G


def write_input_file(G, path):
    with open(path, "w") as fo:
        n = len(G)
        lines = nx.generate_edgelist(G, data=["weight"])
        fo.write(str(n) + "\n")
        fo.writelines("\n".join(lines))
        fo.close()


def read_output_file(path, G):
    """
    Parses and validates an input file

    :param path: str, a path
    :param G: the input graph corresponding to this output
    :return: networkx Graph is the output is well formed, AssertionError thrown otherwise
    """
    with open(path, "r") as fo:
        tokens = fo.readline()
        nodes = set()
        for token in tokens.split():
            assert token.isdigit()
            node = int(token)
            assert 0 <= node < len(G)
            nodes.add(node)
        lines = fo.read().splitlines()
        fo.close()

        # validate edges
        for line in lines:
            tokens = line.split()

            assert len(tokens) == 2
            assert tokens[0].isdigit() and int(tokens[0]) in nodes
            u = int(tokens[0])
            assert tokens[1].isdigit() and int(tokens[1]) in nodes
            v = int(tokens[1])
            assert G.has_edge(u, v)

        T = nx.parse_edgelist(lines, nodetype=int, data=(("weight", float),))
        for (u, v, w) in T.edges(data=True):
            edge_in_G = G.get_edge_data(u, v)
            w["weight"] = edge_in_G["weight"]
        T.add_nodes_from(nodes)

        assert len(T) > 0
        assert utils.is_valid_network(G, T)

        return T


def write_output_file(T, path):
    with open(path, "w") as fo:
        fo.write(" ".join(map(str, T.nodes)) + "\n")
        lines = nx.generate_edgelist(T, data=False)
        fo.writelines("\n".join(lines))
        fo.close()
