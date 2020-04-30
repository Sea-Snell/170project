import os
import json
from scrapeDat import Leaderboard

# with open('leaders1.json', 'r') as f:
# 		leader_board = json.load(f)

team = 'all_my_homies_hate_dino_nuggets'

leaderboard = Leaderboard()

ins = list(os.listdir('inputs'))
outs = list(os.listdir('master_outputs'))
top = [item for item in leaderboard.get_team(team) if item.rank == 1]
print(len(outs), len(ins), len(top))