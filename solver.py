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
	core_set = set([v])
	core_edges = set([])
	dists = {v: 0.0}
	pair_dists = {v: {v: 0.0}}
	while len(core_set) < len(G.nodes):
		# dists = total_tree_dists(build_graph(core_set, core_edges))
		best_score = float('inf')
		best_edge = None
		best_v = None
		other_v = None
		for u in core_set:
			for v in G[u]:
		# for e in G.edges:
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


def prune_leafs_brute(G, non_prunable=[]):
	leafs = get_leafs(G, non_prunable)
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

def prune_leafs_sample(G, n, non_prunable=[]):
	leafs = get_leafs(G, non_prunable)
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

def prune_leafs_greedy(G, non_prunable=[]):
	leafs = get_leafs(G, non_prunable)

	best_score = average_pairwise_distance_fast(G)
	best_G = G
	for leaf in leafs:
		new_G = g_with_removed_leafs(best_G, [leaf])
		score = average_pairwise_distance_fast(new_G)
		if score < best_score:
			best_score = score
			best_G = new_G

	return best_G

def prune_leaves_combo(G, max_leafs=10, non_prunable=[]):
	leafs = get_leafs(G, non_prunable)
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

def min_set(G, v, non_prunable=[]):
	new_G = nx.bfs_tree(G, v)
	for item in new_G.edges:
		new_G[item[0]][item[1]]['weight'] = G[item[0]][item[1]]['weight']
	leafs = get_leafs(new_G, non_prunable)
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

def min_cross_edge(G, A, B):
	min_edge = None
	min_weight = float('inf')
	for u in A:
		for v in G[u]:
			if v in B:
				curr_weight = G[u][v]['weight']
				if curr_weight < min_weight:
					min_weight = curr_weight
					min_edge = (u, v)
				elif curr_weight == min_weight and np.random.rand() < 0.5:
					min_weight = curr_weight
					min_edge = (u, v)
	return min_weight, min_edge



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

def g_overlap(T1, T2):
	intersect = nx.intersection(T1, T2)
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

def get_base_cases(G, non_prunable=[], prune=True):
	if prune:
		prune_f = lambda x: prune_leaves_combo(x, 6, non_prunable)
	else:
		prune_f = None
	all_djkistra_prune, all_cost_djkistra_prune = run_from_all(dijkstra, G, prune_f)
	all_prim_prune, all_cost_prim_prune = run_from_all(prim, G, prune_f)
	all_phuckl3s_prune, all_cost_phuckl3s_prune = run_from_all(phuckl3s_graph, G, prune_f)
	all_min_set, all_cost_min_set = run_from_all(lambda G, v: min_set(G, v, non_prunable), G)
	# all_random, all_cost_random = run_from_all(rand_g, G, lambda x: prune_leaves_combo(x, 6, non_prunable))

	all_G = all_djkistra_prune + all_prim_prune + all_phuckl3s_prune + all_min_set
	all_cost = all_cost_djkistra_prune + all_cost_prim_prune + all_cost_phuckl3s_prune + all_cost_min_set
	prunable = [True for item in all_djkistra_prune + all_prim_prune + all_phuckl3s_prune] + [False for item in all_min_set]
	return all_G, all_cost,  prunable

def genetic_smart_combiner(G, intersection):
	ccs = list(nx.connected_components(intersection))
	non_prunable = []
	for i in range(len(ccs)):
		if len(ccs[i]) > 1:
			non_prunable.append(i)
	meta_G, meta_node_to_real, meta_edge_to_real = create_meta_graph(G, ccs)
	new_base, new_cost, prunable = get_base_cases(meta_G, non_prunable=non_prunable, prune=False)
	new_real = [meta_to_real(G, intersection, item, meta_node_to_real, meta_edge_to_real) for item in new_base]
	for i in range(len(new_real)):
		if prunable[i]:
			new_real[i] = prune_leaves_combo(new_real[i], 6)
	return new_real

def genetic_rand_combiner(G, intersection):
	ccs = list(nx.connected_components(intersection))
	combined = prune_leaves_combo(rand_g_general(G, intersection, ccs, np.random.randint(len(ccs))), 6)
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


def new_generation(G, all_G, all_cost, temp, samples, combiner, mutator, uniform_prob):
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

		intersection = g_overlap(full_G[choice1], full_G[choice2])
		avg_overlap += len(list(intersection.edges)) / min(len(list(full_G[choice1].edges)), len(list(full_G[choice2].edges)))
		combined = combiner(G, intersection)
		combined = list(map(lambda x: mutator(x, max(probs), cum_50, len(all_G)), combined))
		new_gen_G.extend(combined)
		new_gen_cost.extend(list(map(average_pairwise_distance_fast, combined)))
	avg_overlap /= samples
	return new_gen_G, new_gen_cost, avg_overlap, probs

def genetic_loop(G, file, generations, samples, combiner, mutator, temp_timer, initial_Gs, initial_cost, past_keep_percent, all_keep_percent, uniform_prob):
	pop, pop_cost = prune_same(G, initial_Gs, initial_cost)
	print('%s initialziing... Valid: %s, current best: %f, Pop size: %d' % (file, str(is_valid_network(G, pop[np.argmin(pop_cost)])), min(pop_cost), len(pop)))
	for i in range(generations):
		new_pop, new_pop_cost, avg_overlap, probs = new_generation(G, pop, pop_cost, temp_timer(i, len(pop)), samples, combiner, mutator, uniform_prob)
		top_pop = sorted(zip(pop, pop_cost), key=lambda x: x[1])[:max(int(len(pop) * past_keep_percent(len(pop))), 2)]
		best_pop, best_scores = zip(*top_pop)
		pop = new_pop + list(best_pop)
		pop_cost = new_pop_cost + list(best_scores)
		pop, pop_cost = prune_same(G, pop, pop_cost)
		top_pop = sorted(zip(pop, pop_cost), key=lambda x: x[1])[:max(int(len(pop) * all_keep_percent(len(pop))), 2)]
		best_pop, best_scores = zip(*top_pop)
		pop, pop_cost = best_pop, best_scores
		print('%s generation %d done. Valid: %s, Current best: %f, Pop size: %d, Avg overlap: %f, Top 10 Probs: %s, Nodes: %d, Edges: %d' % (file, i, str(is_valid_network(G, pop[np.argmin(pop_cost)])), min(pop_cost), len(pop), avg_overlap, str(sorted(probs)[-10:]), len(list(G.nodes)), len(list(G.edges))))
	return pop, pop_cost

def mutator(G, to_remove, prob):
	if np.random.rand() < prob and len(G.edges) >= to_remove:
		return mutate(G, to_remove)
	return G


def solve(G, size, file):
	"""
	Args:
		G: networkx.Graph

	Returns:
		T: networkx.Graph
	"""

	all_G, all_cost, _ = get_base_cases(G)
	if size == 'small':
		samples = 100
	elif size == 'medium':
		samples = 60
	elif size == 'large':
		samples = 15

	new_G, new_cost = genetic_loop(G, file, generations=10, samples=samples, combiner=genetic_smart_combiner, mutator=mutate_control, temp_timer=temp_timer, initial_Gs=all_G, initial_cost=all_cost, past_keep_percent=lambda x: 1.0, all_keep_percent=keep_percent, uniform_prob=uniform_prob)
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
	return 0.9975 ** num_G + (10 / num_G)

def keep_percent(num_G):
	if num_G > 1000:
		return 0.5
	if num_G < 40:
		return 1.0
	return 0.9

# seed = 20
# random.seed(seed)
# np.random.seed(seed)

# input_item = 'small-8'
# if 'small' in input_item:
# 	size = 'small'
# elif 'medium' in input_item:
# 	size = 'medium'
# elif 'large' in input_item:
# 	size = 'large'
# file = 'inputs/%s.in' % (input_item)
# out_path = './master_outputs'
# out_file = '%s.out' % (input_item)
# testG = read_input_file(file)
# T = solve(testG, size, file)

if __name__ == '__main__':
	assert len(sys.argv) == 4
	in_path = sys.argv[1]
	out_path = sys.argv[2]
	team_name = sys.argv[3]

	seed = 21
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

			T = solve(G, size, in_file)
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

