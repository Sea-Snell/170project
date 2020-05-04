import os
import json
from scrapeDat import Leaderboard
from utils import is_valid_network, average_pairwise_distance, average_pairwise_distance_fast
from parse import read_input_file, write_output_file, read_output_file

# with open('leaders1.json', 'r') as f:
# 		leader_board = json.load(f)

team = 'all_my_homies_hate_dino_nuggets2.0'

leaderboard = Leaderboard()
leaderboard.create_custom_team(team)

in_path = 'inputs'
out_path = 'master_outputs4'
for file in list(os.listdir(out_path)):
	if '.out' in file:
		input_name = file[:-4]
		matching_G = read_input_file(os.path.join(in_path, input_name + '.in'))
		curr_T = read_output_file(os.path.join(out_path, file), matching_G)
		leaderboard.custom_entry(team, input_name, average_pairwise_distance_fast(curr_T), update_leaderboard=False)
leaderboard.update_leaderboard()
top = [item for item in leaderboard.get_team(team) if item.rank == 1]
print(len(top), leaderboard.get_team_rank(team), len(leaderboard.leaderboard))
print(leaderboard.leaderboard)