import networkx as nx

def non_critical_edge_set(G):
	non_critical_edges = []
	for e in G.edges:
		temp = G.copy()
		temp.remove_edge(list(e)[0], list(e)[1])
		if nx.is_connected(temp):
			non_critical_edges.append(tuple(e))
	return non_critical_edges

def brute_fullT(G, memo={}):
	if tuple(G.edges) in memo:
		return memo[tuple(G.edges)]
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
	if best_G != None:
		medium[tuple(G.edges)] = (best_G, best_score)
	return best_G, best_score

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

def prune_leafs_brute(G, non_prunable=[]):
	leafs = get_leafs(G, non_prunable)
	all_combos = powerset(leafs)
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