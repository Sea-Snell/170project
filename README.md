# CS 170 Project Spring 2020
ALL_My_HOMIES_HATE_DINO_NUGGETS algorithm


Ok so to run this just do:
python solver.py path-to-inputs path-to-outputs temperary-team-name

For example you could run
python solver.py inputs/ master_outputs4/ all_my_homies_hate_dino_nuggets_2.0


The team name field doesn't really matter, it just puts that in as your name on a local version of the leaderboard scraped from the web, so it can figure out where your outputs folder will rank. (NOTE: if the database for the leaderboard is down when you run this, it may error, instructions on what to do in that case are below)


The algorithm preferably can take in an outputs folder filled with outputs already, it will run the algorithm and only write over an output if this run improved on it.

In this github we've attached our the current outputs folder "master_outputs4", and older outputs folders are in the "out_archive" folder, so you can run the genetic algorithm starting from any of those folders.


Almost all the code is in solve.py, there are a couple other important files ScrapeDat.py, for scraping the leaderboard, and cluster.py, taken from this github https://github.com/53RT/Highly-Connected-Subgraphs-Clustering-HCS for clustering graphs by highly connected components.


When you run this you may want to try different random seeds, around line 950 in solve.py is where th seed is set, if you want to try to reproduce our results you would want to run seeds (10, 20, 21, 42, 69, 420, and 42069). Although this may not perfectly reproduce, since seeds 10, 20, and 21 were ran on earlier versions of the code, in which case you could go back 1 commit in the github to obtain an earlier version of the algorithm.


And lastly, in the case where the leaderboard database is down but you still want to run it. I wrote another loop that is commited out, look for the commited out if __name__ == '__main__': around line 890 in solve.py and commit out everything below the other if __name__ == '__main__':


