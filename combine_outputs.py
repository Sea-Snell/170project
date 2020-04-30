from parse import read_output_file, write_output_file, read_input_file
from scrapeDat import Leaderboard
import os
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast

input_folder = './inputs/'
folders = ['./outputs_v1/', './outputs_v2/']
merge_folder = './master_outputs/'
leaderboard = Leaderboard()

new_team = 'all_my_homies_hate_dino_nuggets'
leaderboard.create_custom_team(new_team)

for item in leaderboard.input_set:
	G = read_input_file(os.path.join(input_folder, item + '.in'))
	best_score = float('inf')
	best_T = None
	for folder in folders:
		try:
			T = read_output_file(os.path.join(folder, item + '.out'), G)
		except:
			continue
		if is_valid_network(G, T):
			T_score = average_pairwise_distance_fast(T)
			if T_score < best_score:
				best_score = T_score
				best_T = T
	leaderboard.custom_entry(new_team, item, best_score)
	write_output_file(best_T, os.path.join(merge_folder, item + '.out'))

print(leaderboard.get_team_rank(new_team))
print(len(leaderboard.leaderboard))