import networkx as nx
from parse import read_input_file, write_output_file, read_output_file
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
import scipy
import json
from shutil import copyfile
from multiprocessing import Pool
import multiprocessing
from scrapeDat import Leaderboard
from scipy.special import comb
from itertools import chain, combinations
import random_graph
import functools
import cluster

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

def kruskal(G):
	edges = [(item[0], item[1], G[item[0]][item[1]]['weight']) for item in list(nx.minimum_spanning_edges(G, algorithm='kruskal'))]
	T = build_graph(G.nodes, edges)
	return T

def total_tree_dists(G):
	path_lengths = list(nx.all_pairs_dijkstra_path_length(G))
	off_cost = {}
	for lens in path_lengths:
		off_cost[lens[0]] = sum(list(lens[1].values()))
	path_lengths_dict = {item[0]:item[1] for item in path_lengths}
	return off_cost, path_lengths_dict

def phuckl3s(G, v):
	core_set = set([v])
	core_edges = set([])
	dists = {v: 0.0}
	pair_dists = {v: {v: 0.0}}
	while len(core_set) < len(G.nodes):
		best_score = float('inf')
		best_edge = None
		best_v = None
		other_v = None
		for u in core_set:
			for v in G[u]:
				edge_tup = (u, v, G[u][v]['weight'])
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
						other_v = core_e

		core_set.add(best_v)
		core_edges.add(best_edge)
		
		pair_dists[best_v] = {item: pair_dists[other_v][item] + best_edge[2] for item in pair_dists[other_v]}
		pair_dists[best_v][best_v] = 0.0
		dists[best_v] = sum(list(pair_dists[best_v].values()))
		for item in pair_dists:
			pair_dists[item][best_v] = pair_dists[best_v][item]
			dists[item] += pair_dists[best_v][item]



	return core_edges


def phuckl3s_graph(G, v):
	edges = phuckl3s(G, v)
	return build_graph(G.nodes, edges)

def get_leafs(G, non_prunable=[]):
	leafs = []
	for node in G.nodes:
		if G.degree[node] == 1 and int(node) not in non_prunable:
			leafs.append(node)
	return leafs


def get_all_subsets(A):
	subsets = []
	for i in range(len(A) + 1):
		subsets += itertools.combinations(A, i)
	return subsets

def random_sample_from_set(s):
	new_s = []
	for item in s:
		if np.random.rand() < 0.5:
			new_s.append(item)
	return new_s

def is_prunable(G, T, node):
	if not T.has_node(node) or T.degree[node] > 1:
		return False
	for u in G[node]:
		if not T.has_node(u):
			has_other_ajacent = False
			for v in G[u]:
				if T.has_node(v) and v != node:
					has_other_ajacent = True
					break
			if not has_other_ajacent:
				return False
	return True

def prune2(G, T, decider, temp, early_stop, early_stop_prob):
	pruned_T = T.copy()
	prunable = set(filter(lambda node: is_prunable(G, pruned_T, node), list(pruned_T.nodes)))
	pruned = []
	while len(prunable) > 0:
		to_prune = decider(G, pruned_T, prunable, temp, early_stop, early_stop_prob)
		if to_prune == None:
			return pruned_T
		pruned_T.remove_node(to_prune)
		prunable.remove(to_prune)
		pruned.append(to_prune)
		for u in G[to_prune]:
			prunable.add(u)
		prunable = set(filter(lambda node: is_prunable(G, pruned_T, node), list(prunable)))
	return pruned_T

def rand_prune_dumb(G, T, prunable, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	return random.choice(list(prunable))

def smart_prune_initial_state(T):
	totals, path_lengths = total_tree_dists(T)
	n = len(list(T.nodes))
	C = average_pairwise_distance_fast(T) * n * (n - 1) * 0.5
	return totals, path_lengths, n, C

def prune_state(T, smart_decider, totals, path_lengths, n, C):
	totals = dict(totals)
	path_lengths = {item: dict(path_lengths[item]) for item in path_lengths}
	n = int(n)
	C = float(C)

	def decider_f(G, T, prunable, temp, early_stop, early_stop_prob):
		nonlocal totals
		nonlocal path_lengths
		nonlocal n
		nonlocal C

		to_prune = None
		actually_prunable = list(filter(lambda node: totals[node] > ((2 * C) / n), prunable))
		if len(actually_prunable) > 0:
			to_prune = smart_decider(G, T, actually_prunable, list(map(lambda node: totals[node], actually_prunable)), temp, early_stop, early_stop_prob)
			if to_prune == None:
				return None
			for node in list(T.nodes):
				totals[node] -= path_lengths[to_prune][node]
			C -= totals[to_prune]
			n -= 1
		return to_prune

	return decider_f

def greedy_top(G, T, actually_prunable, scores, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	return actually_prunable[np.argmax(scores)]

def greedy_bottom(G, T, actually_prunable, scores, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	return actually_prunable[np.argmin(scores)]

def rand_prune_smart_uniform(G, T, actually_prunable, scores, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	return random.choice(actually_prunable)

def rand_prune_smart_positive(G, T, actually_prunable, scores, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	probs = softmax(np.array(scores) - np.mean(scores), temp=temp)
	return actually_prunable[np.random.choice(len(probs), size=1, p=probs)[0]]

def rand_prune_smart_negative(G, T, actually_prunable, scores, temp, early_stop, early_stop_prob):
	if early_stop and np.random.rand() < early_stop_prob:
		return None
	probs = softmax(-np.array(scores) + np.mean(scores), temp=temp)
	return actually_prunable[np.random.choice(len(probs), size=1, p=probs)[0]]

def smart_decider_set(T, deciders):
	totals, path_lengths, n, C = smart_prune_initial_state(T)
	return [prune_state(T, decider, totals, path_lengths, n, C) for decider in deciders]

def prune2_master(G, T, samples, temp, early_stop_prob):
	totals, path_lengths, n, C = smart_prune_initial_state(T)
	rands = [rand_prune_smart_uniform, rand_prune_smart_positive, rand_prune_smart_negative]
	determinisic = [greedy_top, greedy_bottom]
	early_stop_items = [True, False]
	best_T = T
	Ts = []
	best_score = average_pairwise_distance_fast(T)
	for item in rands:
		for early_stop in early_stop_items:
			for i in range(samples):
				new_T = prune2(G, T, prune_state(T, item, totals, path_lengths, n, C), temp, early_stop, early_stop_prob)
				new_score = average_pairwise_distance_fast(new_T)
				Ts.append(new_T)
				if new_score < best_score:
					best_score = new_score
					best_T = new_T
	for early_stop in early_stop_items:
		for i in range(samples):
			new_T = prune2(G, T, rand_prune_dumb, temp, early_stop, early_stop_prob)
			new_score = average_pairwise_distance_fast(new_T)
			Ts.append(new_T)
			if new_score < best_score:
				best_score = new_score
				best_T = new_T
	for item in determinisic:
		for early_stop in early_stop_items:
			new_T = prune2(G, T, prune_state(T, item, totals, path_lengths, n, C), temp, early_stop, early_stop_prob)
			new_score = average_pairwise_distance_fast(new_T)
			Ts.append(new_T)
			if new_score < best_score:
				best_score = new_score
				best_T = new_T
	return best_T, Ts


def g_with_removed_leafs(G, leafs):
	new_G = G.copy()
	for leaf in leafs:
		new_G.remove_node(leaf)
	return new_G

def get_all_cost(Gs):
	return [average_pairwise_distance_fast(g) for g in Gs]

def display_graph(G):
	plt.clf()
	nx.draw(G)
	plt.show()

def run_from_all(f, G):
	vertices = list(G.nodes)
	all_G = []
	for v in vertices:
		new_G = f(G, v)
		all_G.append(new_G)
	all_cost = get_all_cost(all_G)
	return all_G, all_cost

def min_set(G, v, non_prunable=[]):
	new_G = nx.bfs_tree(G, v).to_undirected()
	for item in new_G.edges:
		new_G[item[0]][item[1]]['weight'] = G[item[0]][item[1]]['weight']
	leafs = get_leafs(new_G, non_prunable)
	if len(leafs) == len(G.nodes):
		return g_with_removed_leafs(new_G, leafs[:1])
	return g_with_removed_leafs(new_G, leafs)

def rand_g(G, v):
	frontier = set()
	new_g = nx.Graph()
	included = set([v])
	for n in G.nodes:
		new_g.add_node(n)
	for u in G[v]:
		frontier.add((v, u))

	while len(frontier) > 0:
		chosen = random.sample(frontier, 1)[0]
		frontier.remove(chosen)
		if chosen[1] in included:
			continue
		new_g.add_edge(chosen[0], chosen[1], weight=G[chosen[0]][chosen[1]]['weight'])
		included.add(chosen[1])
		for u in G[chosen[1]]:
			if u not in included:
				frontier.add((chosen[1], u))
	return new_g

def fully_rand_g(G):
	return rand_g(G, random.choice(list(G.nodes)))

def min_cross_edge(G, A, B, rand_prob=0.01):
	crossing_edges = []
	for u in A:
		for v in G[u]:
			if v in B:
				crossing_edges.append((u, v))

	if len(crossing_edges) == 0:
		return float('inf'), None
	if np.random.rand() < rand_prob:
		e = random.choice(crossing_edges)
		return G[e[0]][e[1]]['weight'], e
	e = min(crossing_edges, key=lambda x: G[x[0]][x[1]]['weight'])
	return G[e[0]][e[1]]['weight'], e



def create_meta_graph(G, groups):
	group_arr = cc_to_group_arr(groups)
	meta_G = nx.Graph()
	meta_node_to_real = {}
	meta_edge_to_real = {}
	for i in range(len(groups)):
		 meta_G.add_node(i)
		 meta_node_to_real[i] = groups[i]

	for i in range(len(groups)):
		for x in range(i + 1, len(groups)):
			min_weight, min_edge = min_cross_edge(G, groups[i], groups[x])
			if min_edge != None:
				meta_G.add_edge(group_arr[min_edge[0]], group_arr[min_edge[1]], weight=min_weight)
				meta_edge_to_real[(group_arr[min_edge[0]], group_arr[min_edge[1]])] = (min_edge[0], min_edge[1])
	return meta_G, meta_node_to_real, meta_edge_to_real

#original G - pre meta full graph
#curr_G - pre meta partial graph to build on
#meta_G - post meta new graph
def meta_to_real(original_G, curr_G, meta_G, meta_node_to_real, meta_edge_to_real):
	new_G = curr_G.copy()
	for edge in meta_G.edges:
		if edge in meta_edge_to_real:
			new_G.add_edge(meta_edge_to_real[edge][0], meta_edge_to_real[edge][1], weight=original_G[meta_edge_to_real[edge][0]][meta_edge_to_real[edge][1]]['weight'])
		else:
			rev_e = (edge[1], edge[0])
			new_G.add_edge(meta_edge_to_real[rev_e][0], meta_edge_to_real[rev_e][1], weight=original_G[meta_edge_to_real[rev_e][0]][meta_edge_to_real[rev_e][1]]['weight'])
	
	real_to_meta = {}
	for m in meta_node_to_real:
		for r in meta_node_to_real[m]:
			real_to_meta[r] = m

	to_remove = []
	for v in new_G.nodes:
		if real_to_meta[v] not in meta_G:
			to_remove.append(v)	
	for item in to_remove:
		new_G.remove_node(item)

	return new_G

def cc_to_group_arr(cc):
	groups = {}
	for i in range(len(cc)):
		for n in cc[i]:
			groups[n] = i
	return groups

def rand_g_general(G, curr_T, cc, start_cc_idx):
	meta_G, meta_node_to_real, meta_edge_to_real = create_meta_graph(G, cc)
	complete_meta_G = rand_g(meta_G, start_cc_idx)
	return meta_to_real(G, curr_T, complete_meta_G, meta_node_to_real, meta_edge_to_real)

def powerset(iterable):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def estimate_search_space(G):
	n_e, n_v = len(list(G.edges)), len(list(G.nodes))
	total = 0
	for i in range(n_v):
		total += comb(n_e, i)
	return n_e, n_v, total

def brute_master(input_name, G, prune_cond=None, early_stop=None):
	T = nx.Graph()
	global_best_score = float('inf')
	global_best_T = None
	steps = 0
	n_e, n_v, search_space = estimate_search_space(G)
	frountier = set()
	print('%s running brute force... edges: %d, nodes: %d, estimated search space: %f' % (input_name, n_e, n_v, search_space))

	@functools.lru_cache(maxsize=10000)
	def brute2(e, n):
		nonlocal global_best_T
		nonlocal global_best_score
		nonlocal T
		nonlocal G
		nonlocal prune_cond
		nonlocal early_stop
		nonlocal steps
		nonlocal frountier
		nonlocal input_name
		nonlocal n_e
		nonlocal n_v
		nonlocal search_space

		steps += 1
		if steps % 10000 == 0:
			print('brute %s current best score: %f, steps: %d, edges: %d, nodes: %d, estimated search space: %f' % (input_name, global_best_score, steps, n_e, n_v, search_space))
		best_score = float('inf')
		best_T = None
		if is_valid_network(G, T):
			best_T = T.copy()
			best_score = average_pairwise_distance_fast(best_T)
			if best_score < global_best_score:
				global_best_score = best_score
				global_best_T = best_T.copy()
		if early_stop != None and early_stop(G, T, global_best_T, global_best_score, steps):
			return best_T, best_score

		possible_edges = set()
		for v in frountier:
			for u in G[v]:
				if not T.has_node(u):
					possible_edges.add((v, u))

		all_edge_sets = powerset(possible_edges)
		for edges in all_edge_sets:
			if early_stop != None and early_stop(G, T, global_best_T, global_best_score, steps):
				return best_T, best_score
			if len(edges) == 0:
				continue
			for v, u in edges:
				if not T.has_node(u):
					T.add_node(u)
				T.add_edge(u, v, weight=G[u][v]['weight'])
			if prune_cond == None or not prune_cond(G, T, global_best_T, global_best_score, steps):
				for v, u in edges:
					if v in frountier:
						frountier.remove(v)
					frountier.add(u)
				new_T, new_score = brute2(tuple(T.edges), tuple(T.nodes))
				if new_score < best_score:
					best_T = new_T.copy()
					best_score = new_score
				for v, u in edges:
					if u in frountier:
						frountier.remove(u)
					frountier.add(v)
			for v, u in edges:
				T.remove_edge(u, v)
			for v, u in edges:
				if T.has_node(u):
					T.remove_node(u)
		return best_T, best_score

	v = list(G.nodes)[0]
	T.add_node(v)
	frountier.add(v)
	brute2(tuple(T.edges), tuple(T.nodes))
	if early_stop != None and early_stop(G, T, global_best_T, global_best_score, steps):
		print('%s brute force found best score %f in %d steps' % (input_name, global_best_score, steps))
		return global_best_T, global_best_score
	frountier.remove(v)
	T.remove_node(v)
	for u in G[v]:
		T.add_node(u)
		frountier.add(u)
		brute2(tuple(T.edges), tuple(T.nodes))
		if early_stop != None and early_stop(G, T, global_best_T, global_best_score, steps):
			print('%s brute force found best score %f in %d steps' % (input_name, global_best_score, steps))
			return global_best_T, global_best_score
		frountier.remove(u)
		T.remove_node(u)

	print('%s brute force found best score %f in %d steps' % (input_name, global_best_score, steps))
	return global_best_T, global_best_score


def best_graph(graphs):
	best_G = None
	best_score = float('inf')
	for G in graphs:
		new_score = average_pairwise_distance_fast(G)
		if new_score < best_score:
			best_score = new_score
			best_G = G
	return best_G

def g_overlap(T1, T2):
	intersect = nx.intersection(T1, T2).copy()
	for e in intersect.edges:
		intersect[e[0]][e[1]]['weight'] = T1[e[0]][e[1]]['weight']
	return intersect

def softmax(items, temp=1):
	return scipy.special.softmax(np.array(items) / temp)

def fill_nodes(G, T):
	new_T = T.copy()
	for n in list(G.nodes):
		if not new_T.has_node(n):
			new_T.add_node(n)
	return new_T

def make_cc_gs(G, ccs):
	Gs = []
	for cc in ccs:
		new_G = nx.Graph()
		for v in cc:
			new_G.add_node(v)
			for u in G[v]:
				new_G.add_edge(v, u, weight=G[v][u]['weight'])
		Gs.append(new_G)
	return Gs

def combine_clusters(original_G, clusters):
	full_G = nx.Graph()
	for n in list(original_G.nodes):
		full_G.add_node(n)
	for g in clusters:
		for e in list(g.edges):
			full_G.add_edge(e[0], e[1], weight=original_G[e[0]][e[1]]['weight'])
	return full_G

def cluster_and_combine(G, samples, temp):
	clustered = cluster.HCS(G.copy())
	ccs = list(nx.connected_components(clustered))
	cluster_Gs = make_cc_gs(clustered, ccs)
	meta, meta_node_to_real, meta_edge_to_real = create_meta_graph(G, ccs)

	base_Gs_all, scores_all = [], []
	for item in cluster_Gs:
		if len(list(item.nodes)) > 2:
			base_Gs, scores = get_base_cases(item, non_prunable=list(item.nodes))
			base_Gs_all.append(base_Gs)
			scores_all.append(scores)
		else:
			base_Gs_all.append([item.copy()])
			if len(list(item.nodes)) == 1:
				scores_all.append([0.0])
			else:
				scores_all.append([average_pairwise_distance_fast(item)])
	meta_Gs, meta_scores = get_base_cases(meta, non_prunable=list(meta.nodes))

	cluster_Ts = []
	new_Gs = []
	for i in range(samples):
		for x in range(len(base_Gs_all)):
			probs = softmax(-np.array(scores_all[x]) + np.mean(scores_all[x]), temp=temp)
			choice = np.random.choice(len(probs), size=1, p=probs)[0]
			cluster_Ts.append(base_Gs_all[x][choice])
		fully_clustered = combine_clusters(G, cluster_Ts)

		probs = softmax(-np.array(meta_scores) + np.mean(meta_scores), temp=temp)
		choice = np.random.choice(len(probs), size=1, p=probs)[0]
		meta_T = meta_Gs[choice]

		combined_G = meta_to_real(G, fully_clustered, meta_T, meta_node_to_real, meta_edge_to_real)
		new_Gs.append(combined_G)
	return new_Gs

def get_base_cases(G, brute_limit=None, cluster_f=None, non_prunable=[], include_rand=False):
	if brute_limit == None:
		brute_limit = float('-inf')
	_, _, total_search_space = estimate_search_space(G)
	if total_search_space < brute_limit:
		brute_result, brute_score = brute_master('base_case', G, prune_f)
		brute_result, brute_score = [brute_result], [brute_score]
	else:
		brute_result, brute_score = [], []

	all_djkistra, all_cost_djkistra = run_from_all(dijkstra, G)
	all_prim, all_cost_prim = run_from_all(prim, G)
	all_phuckl3s, all_cost_phuckl3s = run_from_all(phuckl3s_graph, G)
	all_min_set, all_cost_min_set = run_from_all(lambda G, v: min_set(G, v, non_prunable), G)
	if cluster_f != None:
		all_cluster = cluster_f(G)
		all_cost_cluster = get_all_cost(all_cluster)
	else:
		all_cluster = []
		all_cost_cluster = []
	if include_rand:
		all_random, all_cost_random = run_from_all(rand_g, G)
	else:
		all_random = []
		all_cost_random = []
	kruskal_G = kruskal(G)
	kruskal_cost = average_pairwise_distance_fast(kruskal_G)

	all_G = all_djkistra + all_prim + all_phuckl3s + all_min_set + all_cluster + brute_result + all_random + [kruskal_G]
	all_cost = all_cost_djkistra + all_cost_prim + all_cost_phuckl3s + all_cost_min_set + all_cost_cluster + brute_score + all_cost_random + [kruskal_cost]
	return all_G, all_cost

def genetic_smart_combiner(G, intersection, prune_samples, early_stop_prob, cluster_f=None, brute_limit=None, include_rand=False, **kwargs):
	ccs = list(nx.connected_components(intersection))
	non_prunable = []
	for i in range(len(ccs)):
		if len(ccs[i]) > 1:
			non_prunable.append(i)
	meta_G, meta_node_to_real, meta_edge_to_real = create_meta_graph(G, ccs)
	new_base, new_cost = get_base_cases(meta_G, brute_limit=brute_limit, non_prunable=non_prunable, cluster_f=cluster_f, include_rand=include_rand)
	new_real = [meta_to_real(G, intersection, item, meta_node_to_real, meta_edge_to_real) for item in new_base]
	all_prune = []
	for i in range(len(new_real)):
		best_T, all_Ts = prune2_master(G, new_real[i], samples=prune_samples, early_stop_prob=early_stop_prob, temp=0.6)
		all_prune.extend(all_Ts)
	return all_prune

def genetic_smart_combiner_union(G, unionG, intersection, prune_samples, early_stop_prob, cluster_f=None, brute_limit=None, include_rand=False, **kwargs):
	ccs = list(nx.connected_components(intersection))
	non_prunable = []
	for i in range(len(ccs)):
		if len(ccs[i]) > 1:
			non_prunable.append(i)
	meta_G, meta_node_to_real, meta_edge_to_real = create_meta_graph(unionG, ccs)
	new_base, new_cost = get_base_cases(meta_G, brute_limit=brute_limit, non_prunable=non_prunable, cluster_f=cluster_f, include_rand=include_rand)
	new_real = [meta_to_real(unionG, intersection, item, meta_node_to_real, meta_edge_to_real) for item in new_base]
	all_prune = []
	for i in range(len(new_real)):
		best_T, all_Ts = prune2_master(G, new_real[i], samples=prune_samples, early_stop_prob=early_stop_prob, temp=0.6)
		all_prune.extend(all_Ts)
	return all_prune

def genetic_rand_combiner(G, intersection, prune_samples, early_stop_prob, **kwargs):
	ccs = list(nx.connected_components(intersection))
	combined = prune2_master(G, rand_g_general(G, intersection, ccs, np.random.randint(len(ccs))), samples=prune_samples, early_stop_prob=early_stop_prob, temp=0.6)
	return [combined]

def mutate(G, remove_n):
	to_remove = random.sample(G.edges, remove_n)
	new_G = G.copy()
	for item in to_remove:
		new_G.remove_edge(item[0], item[1])
	ccs = list(nx.connected_components(new_G))
	return rand_g_general(G, new_G, ccs, np.random.randint(len(ccs)))

def prune_same(G, all_G, all_cost):
	i = 0
	# print(len(all_G))
	master_e = list(G.edges)
	claimed = set()
	to_remove = []
	for i in range(len(all_G)):
		code = []
		for e in master_e:
			code.append(all_G[i].has_edge(e[0], e[1]))
		code = tuple(code)
		if code in claimed:
			to_remove.append(i)
		else:
			claimed.add(code)

	for item in to_remove[::-1]:
		all_G.pop(item)
		all_cost.pop(item)
	# print(len(all_G))
	return all_G, all_cost

def union_nodes(T1, T2, G):
	new_T1 = T1.copy()
	new_T2 = T2.copy()
	new_G = nx.Graph()
	all_nodes = set(T1.nodes).union(set(T2.nodes))
	for n in all_nodes:
		if not new_T1.has_node(n):
			new_T1.add_node(n)
		if not new_T2.has_node(n):
			new_T2.add_node(n)
		if not new_G.has_node(n):
			new_G.add_node(n)
	for e in list(G.edges):
		if new_G.has_node(e[0]) and new_G.has_node(e[1]):
			new_G.add_edge(e[0], e[1], weight=G[e[0]][e[1]]['weight'])
	return new_T1, new_T2, new_G

def new_generation(G, all_G, all_cost, temp, samples, combiner, mutator, uniform_prob, cluster_f, brute_limit, include_rand, prune_samples, early_stop_prob):
	all_cost_logit = -np.array(all_cost)
	all_cost_logit -= np.mean(all_cost_logit)
	probs = softmax(all_cost_logit, temp=temp)
	cum_50 = 0
	cum = 0.0
	for item in sorted(probs, reverse=True):
		cum += item
		cum_50 += 1
		if cum > 0.5:
			break

	full_G = [fill_nodes(G, g) for g in all_G]

	new_gen_G = []
	new_gen_cost = []
	avg_overlap = 0.0
	for i in range(samples):
		if np.random.rand() < uniform_prob(max(probs), cum_50, len(all_G)):
			choice1, choice2 = random.sample(list(range(len(probs))), 2)
		else:
			choice1 = np.random.choice(len(probs), size=1, p=probs)[0]
			temp = list(probs)
			probs[choice1] = 0.0
			probs /= np.sum(probs)
			choice2 = np.random.choice(len(probs), size=1, p=probs)[0]
			probs = temp

		unionT1, unionT2, unionG = union_nodes(all_G[choice1], all_G[choice2], G)
		intersection_union = g_overlap(unionT1, unionT2)
		intersection = g_overlap(full_G[choice1], full_G[choice2])
		avg_overlap += len(list(intersection.edges)) / min(len(list(full_G[choice1].edges)), len(list(full_G[choice2].edges)))
		avg_overlap += len(list(intersection_union.edges)) / min(len(list(unionT1.edges)), len(list(unionT2.edges)))
		combined = combiner(G, intersection, brute_limit=brute_limit, cluster_f=cluster_f, include_rand=include_rand, prune_samples=prune_samples, early_stop_prob=early_stop_prob)
		combined_union = genetic_smart_combiner_union(G, unionG, intersection_union, brute_limit=brute_limit, cluster_f=cluster_f, include_rand=include_rand, prune_samples=prune_samples, early_stop_prob=early_stop_prob)
		combined = list(map(lambda x: mutator(x, max(probs), cum_50, len(all_G)), combined))
		combined_union = list(map(lambda x: mutator(x, max(probs), cum_50, len(all_G)), combined))
		new_gen_G.extend(combined)
		new_gen_G.extend(combined_union)
		new_gen_cost.extend(list(map(average_pairwise_distance_fast, combined)))
		new_gen_cost.extend(list(map(average_pairwise_distance_fast, combined_union)))
	avg_overlap /= (2 * samples)
	return new_gen_G, new_gen_cost, avg_overlap, probs

def genetic_loop(G, file, generations, samples, combiner, mutator, temp_timer, initial_Gs, initial_cost, past_keep_percent, all_keep_percent, uniform_prob, cluster_f, brute_limit, top_score, prune_samples, early_stop_prob):
	pop, pop_cost = prune_same(G, initial_Gs, initial_cost)
	print('%s initialziing... Valid: %s, current best: %f, Pop size: %d' % (file, str(is_valid_network(G, pop[np.argmin(pop_cost)])), min(pop_cost), len(pop)))
	if is_top_score(min(pop_cost), top_score):
		print('%s found top score %f' % (file, min(pop_cost)))
		return pop, pop_cost
	for i in range(generations):
		if len(pop) < 20:
			include_rand = True
		else:
			include_rand = False
		new_pop, new_pop_cost, avg_overlap, probs = new_generation(G, pop, pop_cost, temp_timer(i, len(pop)), samples, combiner, mutator, uniform_prob, cluster_f(i, len(pop)), brute_limit, include_rand, prune_samples, early_stop_prob)
		top_pop = sorted(zip(pop, pop_cost), key=lambda x: x[1])[:max(int(len(pop) * past_keep_percent(len(pop))), 2)]
		best_pop, best_scores = zip(*top_pop)
		pop = new_pop + list(best_pop)
		pop_cost = new_pop_cost + list(best_scores)
		pop, pop_cost = prune_same(G, pop, pop_cost)
		top_pop = sorted(zip(pop, pop_cost), key=lambda x: x[1])[:max(int(len(pop) * all_keep_percent(len(pop))), 2)]
		best_pop, best_scores = zip(*top_pop)
		pop, pop_cost = best_pop, best_scores
		print('%s generation %d done. Valid: %s, Current best: %f, Pop size: %d, Avg overlap: %f, Top 10 Probs: %s, Nodes: %d, Edges: %d' % (file, i, str(is_valid_network(G, pop[np.argmin(pop_cost)])), min(pop_cost), len(pop), avg_overlap, str(sorted(probs)[-10:]), len(list(G.nodes)), len(list(G.edges))))
		if is_top_score(min(pop_cost), top_score):
			print('%s found top score %f' % (file, min(pop_cost)))
			return pop, pop_cost
	return pop, pop_cost

def mutator(G, to_remove, prob):
	if np.random.rand() < prob and len(G.edges) >= to_remove:
		return mutate(G, to_remove)
	return G


def solve(G, file, top_score):
	"""
	Args:
		G: networkx.Graph

	Returns:
		T: networkx.Graph
	"""
	if 'small' in file:
		size = 'small'
	elif 'medium' in file:
		size = 'medium'
	elif 'large' in file:
		size = 'large'
	brute_limit = float('-inf')
	early_stop_prob = 0.5
	prune_samples = 1

	all_G, all_cost = get_base_cases(G, brute_limit=brute_limit, cluster_f=cluster_f_wrapper(-1, -1), include_rand=True)
	all_prune_G = []
	for new_G in all_G:
		all_prune_G.extend(prune2_master(G, new_G, samples=prune_samples, temp=0.6, early_stop_prob=early_stop_prob)[1])
	all_cost = get_all_cost(all_prune_G)
	if size == 'small':
		samples = 30
	elif size == 'medium':
		samples = 15
	elif size == 'large':
		samples = 3

	new_G, new_cost = genetic_loop(G, file, generations=7, samples=samples, combiner=genetic_smart_combiner, mutator=mutate_control, temp_timer=temp_timer, initial_Gs=all_prune_G, initial_cost=all_cost, past_keep_percent=lambda x: 1.0, all_keep_percent=keep_percent, uniform_prob=uniform_prob, cluster_f=cluster_f_wrapper, brute_limit=brute_limit, top_score=top_score, prune_samples=prune_samples, early_stop_prob=early_stop_prob)
	print('%s without evolution best %f' % (file, min(all_cost)))
	print('%s with evolution best %f' % (file, min(new_cost)))
	best_G = new_G[np.argmin(new_cost)]
	return best_G

def uniform_prob(top_prob, num_50, num_G):
	if top_prob > 0.8:
		return 0.5
	if num_50 / num_G < 0.01:
		return 0.333
	return 0.15

def mutate_control(G, top_prob, num_50, num_G):
	if top_prob > 0.85:
		return mutator(G, np.random.randint(1, 6), 0.04)
	return mutator(G, np.random.randint(1, 4), 0.02)

def temp_timer(x, num_G):
	if num_G > 1500:
		return 0.9975 ** (1500) + (10 / 1500)
	return 0.9975 ** num_G + (10 / num_G)

def cluster_f_wrapper(x, num_G):
	def cluster_f(G):
		samples = len(list(G.nodes))
		if x == -1:
			temp = 1.0
		else:
			temp = temp_timer(x, num_G)
		return cluster_and_combine(G, samples, temp)
	return cluster_f

def keep_percent(num_G):
	if num_G > 2500:
		return 0.4
	if num_G > 1000:
		return 0.5
	if num_G < 40:
		return 1.0
	return 0.9

def prune_f(G, T, global_best_T, global_best_score, steps):
	if len(list(T.edges)) > 0.5 * (len(list(G.nodes)) - 1) and average_pairwise_distance_fast(T) * 0.5 > global_best_score:
		# print('pruned', global_best_score, average_pairwise_distance_fast(T), steps)
		return True
	return False


def early_stop_wrapper(best_score):
	def early_stop_f(G, T, global_best_T, global_best_score, steps):
		return is_top_score(global_best_score, best_score)
	return early_stop_f

def is_top_score(score, top_score):
	return score - top_score <= 1e-9


# if __name__ == '__main__':
# 	assert len(sys.argv) == 4
# 	in_path = sys.argv[1]
# 	out_path = sys.argv[2]
# 	team_name = sys.argv[3]

# 	input_sequence = [for item in os.listdir(in_path) if '.in' in item]

# 	def process_input(input_name):
# 		size = None
# 		if 'small' in input_name:
# 			size = 'small'
# 		elif 'medium' in input_name:
# 			size = 'medium'
# 		elif 'large' in input_name:
# 			size = 'large'

# 		out_file = input_name + '.out'
# 		in_file = input_name + '.in'
# 		print('starting %s...' % (input_name))
# 		G = read_input_file(os.path.join(in_path, in_file))
# 		try:
# 			curr_T = read_output_file(os.path.join(out_path, out_file), G)
# 			if is_valid_network(G, curr_T):
# 				starting_score = average_pairwise_distance_fast(curr_T)
# 			else:
# 				print('bad network %s' % (input_name))
# 				curr_T = None
# 				starting_score = None
# 		except:
# 			print('no current output %s' % (input_name))
# 			curr_T = None
# 			starting_score = None

# 		print('input %s: current score = %s' % (input_name, str(starting_score)))

# 		T = solve(G, in_file, 0.0)
# 		print('done %s' % (input_name))
# 		assert is_valid_network(G, T)
# 		new_score = average_pairwise_distance_fast(T)
# 		if starting_score == None or new_score < starting_score:
# 			print('inputs %s: old score = %s, new score = %s' % (input_name, str(starting_score), str(new_score)))
# 			print('saving result...')
# 			write_output_file(T, os.path.join(out_path, out_file))
# 			print('saved.')
# 		else:
# 			print('%s old score was better. old: %s, new: %s' % (input_name, str(starting_score), str(new_score)))

# 	def init_lock(l):
# 		global lock
# 		lock = l

# 	l = multiprocessing.Lock()
# 	pool = Pool(os.cpu_count(), initializer=init_lock, initargs=(l,))
# 	pool.map(process_input, input_sequence)

if __name__ == '__main__':
	assert len(sys.argv) == 4
	in_path = sys.argv[1]
	out_path = sys.argv[2]
	team_name = sys.argv[3]

	seed = 42069
	random.seed(seed)
	np.random.seed(seed)

	print('setting up leaderboard...')
	leaderboard = Leaderboard()
	leaderboard.create_custom_team(team_name)

	un_seen_inputs = set(leaderboard.input_set)

	for file in list(os.listdir(out_path)):
		if '.out' in file:
			input_name = file[:-4]
			matching_G = read_input_file(os.path.join(in_path, input_name + '.in'))
			curr_T = read_output_file(os.path.join(out_path, file), matching_G)
			if is_valid_network(matching_G, curr_T):
				leaderboard.custom_entry(team_name, input_name, average_pairwise_distance_fast(curr_T), update_leaderboard=False)
				un_seen_inputs.remove(input_name)
	leaderboard.update_leaderboard()
	initial_rank = leaderboard.get_team_rank(team_name)
	print('done.')
	print('starting all rank: %s' % (str(initial_rank)))

	input_sequence = list(un_seen_inputs) + list(map(lambda x: x.input_name, sorted(leaderboard.get_team(team_name), key=lambda x: x.rank, reverse=True)))

	def process_input(input_name):
		size = None
		if 'small' in input_name:
			size = 'small'
		elif 'medium' in input_name:
			size = 'medium'
		elif 'large' in input_name:
			size = 'large'

		out_file = input_name + '.out'
		in_file = input_name + '.in'
		lock.acquire()
		starting_rank = leaderboard.get_rank(team_name, input_name)
		leaderboard_top_score = leaderboard.get_top_score(input_name)
		lock.release()
		if starting_rank != 1:
			print('starting %s...' % (input_name))
			G = read_input_file(os.path.join(in_path, in_file))
			try:
				curr_T = read_output_file(os.path.join(out_path, out_file), G)
				if is_valid_network(G, curr_T):
					starting_score = average_pairwise_distance_fast(curr_T)
				else:
					print('bad network %s' % (input_name))
					curr_T = None
					starting_rank = None
					starting_score = None
			except:
				print('no current output %s' % (input_name))
				curr_T = None
				starting_rank = None
				starting_score = None

			print('input %s: current score = %s, current rank = %s' % (input_name, str(starting_score), str(starting_rank)))

			T = solve(G, in_file, leaderboard_top_score)
			print('done %s' % (input_name))
			assert is_valid_network(G, T)
			new_score = average_pairwise_distance_fast(T)
			if starting_score == None or new_score < starting_score:
				lock.acquire()
				leaderboard.custom_entry(team_name, input_name, new_score)
				new_all_rank = leaderboard.get_team_rank(team_name)
				new_rank = leaderboard.get_rank(team_name, input_name)
				lock.release()
				print('inputs %s: old score = %s, new score = %s, old rank = %s, new rank = %s' % (input_name, str(starting_score), str(new_score), str(starting_rank), str(new_rank)))
				print('all rank from %s to %s' % (str(initial_rank), str(new_all_rank)))
				print('saving result...')
				write_output_file(T, os.path.join(out_path, out_file))
				print('saved.')
			else:
				print('%s old score was better. old: %s, new: %s' % (input_name, str(starting_score), str(new_score)))
		else:
			print('skipping %s cause already rank 1' % (input_name))

	def init_lock(l):
		global lock
		lock = l

	l = multiprocessing.Lock()
	pool = Pool(os.cpu_count(), initializer=init_lock, initargs=(l,))
	pool.map(process_input, input_sequence)

