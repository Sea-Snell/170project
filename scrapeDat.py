from firebase import firebase
from collections import defaultdict
import numpy as np
from functools import total_ordering

@total_ordering
class Entry:
	def __init__(self, team_name, input_name, score, rank=None):
		self.team_name = team_name
		self.input_name = input_name
		self.score = round(score, 8)
		self.rank = rank

	def __str__(self):
		return "team: %s, input: %s, score: %f, rank: %s" % (self.team_name, self.input_name, self.score, str(self.rank))

	def __repr__(self):
		return "{\'team_name\': %s, \'input_name\': %s, \'score\': %f, \'rank\': %s}" % (self.team_name, self.input_name, self.score, str(self.rank))

	def __eq__(self, other):
		return self.score == other.score

	def __ne__(self, other):
		return self.score != other.score

	def __lt__(self, other):
		return self.score < other.score

class Leaderboard:
	def __init__(self):
		self.database = firebase.FirebaseApplication('https://cs-170-project-sp20.firebaseio.com', None)
		self.custom_teams = set()
		self.custom_entries = {}
		self.all_items = {}
		self.team_set = set()
		self.input_set = set()
		self.team_map = defaultdict(list)
		self.input_map = defaultdict(list)
		self.leaderboard = []
		self.refresh()

	def refresh(self):
		dat = self.database.get('/leaderboard', None)
		self.all_items = dict(self.custom_entries)
		self.team_set = set(self.custom_teams)
		self.input_set = set()
		self.team_map = defaultdict(list)
		self.input_map = defaultdict(list)
		for item in dat:
			score = dat[item]['score']
			input_name = dat[item]['input']
			team_name = dat[item]['leaderboard_name']
			info = Entry(team_name, input_name, score)
			self.all_items[(team_name, input_name)] = info
			self.team_set.add(team_name)
			self.input_set.add(input_name)
			self.team_map[team_name].append(info)
			self.input_map[input_name].append(info)

		self.update_leaderboard()

	def create_custom_team(self, team_name):
		assert team_name not in self.team_set and team_name not in self.custom_teams
		self.custom_teams.add(team_name)
		self.team_set.add(team_name)

	def custom_entry(self, team_name, input_name, score, update_leaderboard=True):
		assert input_name in self.input_set and team_name in self.team_set and team_name in self.custom_teams
		if (team_name, input_name) in self.all_items:
			self.all_items[(team_name, input_name)].score = round(score, 8)
		else:
			new_entry = Entry(team_name, input_name, score)
			self.custom_entries[(team_name, input_name)] = new_entry
			self.all_items[(team_name, input_name)] = new_entry
			self.team_map[team_name].append(new_entry)
			self.input_map[input_name].append(new_entry)
		if update_leaderboard:
			self.update_leaderboard()

	def update_leaderboard(self):
		for input_name in self.input_map:
			curr = 0
			prev = -1
			self.input_map[input_name].sort()
			for i in range(len(self.input_map[input_name])):
				if self.input_map[input_name][i].score != prev:
					curr = i + 1
					prev = self.input_map[input_name][i].score
				self.input_map[input_name][i].rank = curr

		self.leaderboard = []
		for team_name in self.team_map:
			self.leaderboard.append((team_name, np.mean(list(map(lambda x: x.rank, self.team_map[team_name])))))
		self.leaderboard.sort(key=lambda x: x[1])

	def get_rank(self, team_name, input_name):
		if (team_name, input_name) in self.all_items:
			return self.all_items[(team_name, input_name)].rank

	def get_scores(self, input_name):
		return self.input_map[input_name]

	def get_team(self, team_name):
		return self.team_map[team_name]

	def get_team_rank(self, team_name):
		for i in range(len(self.leaderboard)):
			if self.leaderboard[i][0] == team_name:
				return i + 1

	def get_entry(self, team_name, input_name):
		return self.all_items[(team_name, input_name)]

	def get_top_score(self, input_name):
		return min(self.input_map[input_name]).score

# test = Leaderboard()
# print(test.leaderboard)
# # # print(test.leaderboard)
# test.create_custom_team('hello_jelani')
# test.custom_entry('hello_jelani', 'small-1', 32.0)
# print(test.get_rank('hello_jelani', 'small-1'))
# # # # print(test.get_rank('hello_jelani', 'small-1'))
# # # # print(test.get_team_rank('hello_jelani'))
# test.custom_entry('hello_jelani', 'small-1', 0.0)
# print(test.get_rank('hello_jelani', 'small-1'))
# print(test.get_team_rank('hello_jelani'))
# print(test.get_rank('jelani_we_love_u', 'small-1'))
# print(test.get_rank('hello_jelani', 'small-1'))
# print(test.get_entry('jelani_we_love_u', 'small-1'))
# print(test.get_entry('hello_jelani', 'small-1'))
# print(test.get_scores('small-1'))
# print(list(test.all_items)[0] in test.all_items)


		

