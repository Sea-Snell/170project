import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
import heapq
import sys
from itertools import count
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import random_graph
import os

def build_graph(vertices, edges):
	G = nx.Graph()
	for v in vertices:
		G.add_node(v)

	for e in edges:
		G.add_edge(e[0], e[1], weight=e[2])

	return G

def dijkstra(G, v):
	paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra_path(G, v)
	return paths_to_graph(G, paths)

def paths_to_graph(G, paths):
	edges = set()
	for node in paths:
		for i in range(1, len(paths[node])):
			edges.add((paths[node][i - 1], paths[node][i], G[paths[node][i - 1]][paths[node][i]]['weight']))

	T = build_graph(G.nodes, edges)
	return T


# modified from network X source code
# https://networkx.github.io/documentation/networkx-1.9/_modules/networkx/algorithms/mst.html
def prim_mst_edges(G, start_node, weight='weight', data=True):
	nodes = G.nodes()
	c = count()

	u = start_node
	frontier = []
	visited = [u]
	for u, v in G.edges(u):
		heapq.heappush(frontier, (G[u][v].get(weight, 1), next(c), u, v))

	while frontier:
		W, _, u, v = heapq.heappop(frontier)
		if v in visited:
			continue
		visited.append(v)

		for v, w in G.edges(v):
			if not w in visited:
				heapq.heappush(frontier, (G[v][w].get(weight, 1), next(c), v, w))
		if data:
			yield u, v, G[u][v]
		else:
			yield u, v

def prim(G, v):
	edges = [(item[0], item[1], item[2]['weight']) for item in list(prim_mst_edges(G, v))]
	T = build_graph(G.nodes, edges)
	return T

def total_tree_dists(G):
	path_lengths = nx.all_pairs_dijkstra_path_length(G)
	off_cost = {}
	for lens in path_lengths:
		off_cost[lens[0]] = sum(list(lens[1].values()))

	return off_cost

def phuckl3s(G, v):
	core_set = [v]
	core_edges = []
	while len(core_set) < len(G.nodes):
		dists = total_tree_dists(build_graph(core_set, core_edges))
		best_score = float('inf')
		best_edge = None
		best_v = None
		for e in G.edges:
			edge_tup = (tuple(e)[0], tuple(e)[1], G[tuple(e)[0]][tuple(e)[1]]['weight'])
			if edge_tup not in core_edges:
				if edge_tup[0] in core_set and edge_tup[1] not in core_set:
					core_e = edge_tup[0]
					non_core_e = edge_tup[1]
				elif edge_tup[0] not in core_set and edge_tup[1] in core_set:
					non_core_e = edge_tup[0]
					core_e = edge_tup[1]
				else:
					continue

				score = dists[core_e] + len(core_set) * edge_tup[2]
				if score < best_score:
					best_score = score
					best_edge = edge_tup
					best_v = non_core_e

		core_set.append(best_v)
		core_edges.append(best_edge)

	return core_edges


def phuckl3s_graph(G, v):
	edges = phuckl3s(G, v)
	return build_graph(G.nodes, edges)

def get_leafs(G):
	leafs = []
	for node in G.nodes:
		if G.degree[node] == 1:
			leafs.append(node)
	return leafs


def get_all_subsets(A):
	subsets = []
	for i in range(len(A) + 1):
		subsets += itertools.combinations(A, i)
	return subsets


def prune_leafs_brute(G):
	leafs = get_leafs(G)
	all_combos = get_all_subsets(leafs)
	if len(leafs) == len(G.nodes):
		all_combos = [combo for combo in all_combos if len(combo) < len(leafs)]
	best_score = float('inf')
	best_G = None
	for subset in all_combos:
		new_G = g_with_removed_leafs(G, subset)

		score = average_pairwise_distance_fast(new_G)
		if score < best_score:
			best_score = score
			best_G = new_G
	return best_G

def random_sample_from_set(s):
	new_s = []
	for item in s:
		if np.random.rand() < 0.5:
			new_s.append(item)
	return new_s

def prune_leafs_sample(G, n):
	leafs = get_leafs(G)
	combos_to_try = [random_sample_from_set(leafs) for i in range(n)]
	if len(leafs) == len(G.nodes):
		combos_to_try = [combo for combo in combos_to_try if len(combo) < len(leafs)]
	best_score = average_pairwise_distance_fast(G)
	best_G = G
	for subset in combos_to_try:
		new_G = g_with_removed_leafs(G, subset)
		score = average_pairwise_distance_fast(new_G)
		if score < best_score:
			best_score = score
			best_G = new_G
	return best_G

def prune_leafs_greedy(G):
	leafs = get_leafs(G)

	best_score = average_pairwise_distance_fast(G)
	best_G = G
	for leaf in leafs:
		new_G = g_with_removed_leafs(best_G, [leaf])
		score = average_pairwise_distance_fast(new_G)
		if score < best_score:
			best_score = score
			best_G = new_G

	return best_G

def prune_leaves_combo(G, max_leafs=10):
	leafs = get_leafs(G)
	if len(leafs) <= max_leafs:
		return prune_leafs_brute(G)
	greedy = prune_leafs_greedy(G)
	sample = prune_leafs_sample(G, max_leafs)
	if average_pairwise_distance_fast(greedy) < average_pairwise_distance_fast(sample):
		return greedy
	return sample


def g_with_removed_leafs(G, leafs):
	new_v = []
	for v in G.nodes:
		if v not in leafs:
			new_v.append(v)
	new_e = []
	for e in G.edges:
		if e[0] not in leafs and e[1] not in leafs:
			new_e.append((e[0], e[1], G[e[0]][e[1]]['weight']))
	return build_graph(new_v, new_e)



def get_all_cost(Gs):
	return [average_pairwise_distance_fast(g) for g in Gs]

def display_graph(G):
	plt.clf()
	nx.draw(G)
	plt.show()

def run_from_all(f, G, prune_f=None):
	vertices = G.nodes.keys()
	all_G = []
	for v in vertices:
		# print('running ' + str(f))
		new_G = f(G, v)
		# print('pruning')
		if prune_f != None:
			new_G = prune_f(new_G)
		# print('done')
		all_G.append(new_G)
	all_cost = get_all_cost(all_G)
	return all_G, all_cost

def min_set(G, v):
	new_G = nx.bfs_tree(G, v)
	for item in new_G.edges:
		new_G[item[0]][item[1]]['weight'] = G[item[0]][item[1]]['weight']
	leafs = get_leafs(new_G)
	if len(leafs) == len(G.nodes):
		return prune_leafs_brute(new_G)
	return g_with_removed_leafs(new_G, leafs)

def non_critical_edge_set(G):
	non_critical_edges = []
	for e in G.edges:
		temp = G.copy()
		temp.remove_edge(list(e)[0], list(e)[1])
		if nx.is_connected(temp):
			non_critical_edges.append(tuple(e))
	return non_critical_edges

def rand_g(G):
	T = G.copy()
	while not is_valid_network(G, T):
		non_critical_edges = non_critical_edge_set(T)
		to_remove = non_critical_edges[np.random.choice(range(len(non_critical_edges)))]
		T.remove_edge(to_remove[0], to_remove[1])
	return T

# def random_dfs(G, v, paths, visited=set()):
# 	visited.add(v)
# 	edges = set(map(lambda x: x[1], G.edges(v))).difference(visited)
# 	if len(edges) == 0:
# 		print('end %d' % (v))
# 		return
# 	u = np.random.choice(list(edges))
# 	paths[u] = paths[v] + [u]
# 	random_dfs(G, u, paths, visited)

# def rand_g2(G, v):
# 	paths = {node: [v] for node in G.nodes}
# 	random_dfs(G, v, paths)
# 	for path in paths:
# 		if len(paths[path]) == 1:
# 			print(path, paths[path])
# 	return paths_to_graph(G, paths)

def brute(G):
	best_G = None
	best_score = float('inf')
	non_critical_edges = non_critical_edge_set(G)
	if len(non_critical_edges) == 0:
		return G, average_pairwise_distance_fast(G)
	for edge in non_critical_edges:
		temp_G = G.copy()
		temp_G.remove_edge(edge[0], edge[1])
		new_G, new_score = brute(temp_G)
		if new_score < best_score:
			best_score = new_score
			best_G = new_G
	return best_G, best_score


def best_graph(graphs):
	best_G = None
	best_score = float('inf')
	for G in graphs:
		new_score = average_pairwise_distance_fast(G)
		if new_score < best_score:
			best_score = new_score
			best_G = G
	return best_G

def solve(G):
	"""
	Args:
		G: networkx.Graph

	Returns:
		T: networkx.Graph
	"""

	all_djkistra_prune, all_cost_djkistra_prune = run_from_all(dijkstra, G, lambda x: prune_leaves_combo(x, 6))
	all_prim_prune, all_cost_prim_prune = run_from_all(prim, G, lambda x: prune_leaves_combo(x, 6))
	all_phuckl3s_prune, all_cost_phuckl3s_prune = run_from_all(phuckl3s_graph, G, lambda x: prune_leaves_combo(x, 6))
	all_min_set, all_cost_min_set = run_from_all(min_set, G)
	# print('random')
	# all_random = [rand_g(G) for i in range(1)]
	# all_cost_random = get_all_cost(all_random)
	# print('done random')

	best_G = best_graph(all_djkistra_prune + all_prim_prune + all_phuckl3s_prune + all_min_set)

	print("dijkstra: %f, prim: %f, phuckl3s: %f, min_set: %f" % (min(all_cost_djkistra_prune), min(all_cost_prim_prune), min(all_cost_phuckl3s_prune), min(all_cost_min_set)))

	return best_G

testG = read_input_file('inputs/medium-112.in')

display_graph(rand_g2(testG, 0))
# print(average_pairwise_distance_fast(rand_g2(testG, 0)))


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
# 	assert len(sys.argv) == 3
# 	in_path = sys.argv[1]
# 	out_path = sys.argv[2]
# 	for file in os.listdir(in_path):
# 		if '.in' in file:
# 			print('running ' + file + '...')
# 			G = read_input_file(in_path + '/' + file)
# 			T = solve(G)
# 			assert is_valid_network(G, T)
# 			print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
# 			write_output_file(T, out_path + '/' + file[:-3] + '.out')

