import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils
import parse

eps = 1e-9
# np.random.seed(0)


def generate_graph(n, edge_prob=0.5):
	"""
	Args:
		n: number of nodes in graph

	Returns:
		G: networkx.Graph
	"""

	G = nx.Graph()

	for i in range(n):
		G.add_node(i)

	for i in range(n):
		for x in range(i+1, n):
			if np.random.rand() < edge_prob:
				w = np.random.randint(1, 100*1000)/1000
				G.add_edge(i, x, weight=w)

	return G

def generate_connected_graph(n, edge_prob=0.5):
	G = generate_graph(n, edge_prob)
	while not nx.is_connected(G):
		G = generate_graph(n, edge_prob)
	return G

def display_graph(G):
	plt.clf()
	nx.draw(G)
	plt.show()

def get_avg_degree(G):
	return 2.0 * len(G.edges) / len(G.nodes)

def write_graph(G, path):
	with open(path, "w+") as f:
		f.write(str(len(G.nodes)) + '\n')
		for edge in G.edges:
			f.write(' '.join(map(str, [edge[0], edge[1], G[edge[0]][edge[1]]['weight']])) + '\n')
		f.close()

# n = 50
# G = generate_connected_graph(n, edge_prob=(np.log(n) / n) + 0.05)
# print(get_avg_degree(G))
# print(nx.is_connected(G))
# display_graph(G)
# write_graph(G, './test')

# G = parse.read_input_file('./25.in')
# print(len(G.nodes), nx.is_connected(G))




