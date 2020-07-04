#!/usr/bin/env python
# # -*- coding: utf-8 -*-
"""
Code: UPC++ - Homework 3
Author: Vlachas Pantelis (pvlachas@ethz.ch)
ETH Zuerich - HPCSE II (Spring 2020)
"""
#!/usr/bin/env python

#############################################################################################
# Function to plot the results of the runs
# Run with command: python3 plot.py --NUM_RANKS 24 --strategy divide_and_conquer
# Run with command: python3 plot.py --NUM_RANKS 24 --strategy producer_consumer
#############################################################################################
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser

def main():
    # GETTING THE NUMBER OF RANKS (USED TO RUN THE CODE)
    parser = ArgumentParser()
    parser.add_argument("-N", "--NUM_RANKS", dest="NUM_RANKS", type=int, help="number of ranks", required=True)
    parser.add_argument("-S", "--strategy", dest="strategy", type=str, help="the strategy, either divide_and_conquer or producer_consumer", required=True)
    args = parser.parse_args()
    NUM_RANKS = args.NUM_RANKS
    strategy = args.strategy
    strategies = [strategy]
    # strategies = ["divide_and_conquer", "producer_consumer"]

    # Make figures directory if it does not exist
    os.makedirs("./Figures", exist_ok=True)

    if not os.path.isdir("./Results"):
        raise ValueError("./Results directory does not exist. Did you created it before running the UPCXX programmes? Did you run the programmes?")


    filename_base = "./Results/{:}_time_rank_{:}.txt"
    for i in range(len(strategies)):
        strategy = strategies[i]
        title = strategy.replace("_", " ")
        rank_times = []
        for rank_id in range(NUM_RANKS):
            filetemp = filename_base.format(strategy, rank_id)
            timetemp = np.loadtxt(filetemp)
            rank_times.append(timetemp)

        rank_times = np.array(rank_times)
        # print(rank_times)
        plt.bar(np.arange(len(rank_times)), height=rank_times)
        plt.title("Histogram of rank times for {:} tasking".format(title))
        plt.ylabel("Time [s]")
        # plt.xticks(np.arange(len(rank_times)))
        plt.xlabel("Rank id")
        # plt.show()
        plt.tight_layout()
        fig_path = "./Figures/Ranktimes_histogram_{:}.png".format(strategy)
        plt.savefig(fig_path)
        plt.close()

    filename_base = "./Results/{:}.txt"

    time_sequential = np.loadtxt(filename_base.format("sequential"), delimiter=",")[1]

    strategies = ["divide_and_conquer", "producer_consumer"]
    for strategy in strategies:
        time = float(np.loadtxt(filename_base.format(strategy), delimiter=",")[1])
        # Computing speed-up
        speed_up = time_sequential / time
        print("Strategy: {:} - Speedup={:}".format(strategy, speed_up))




if __name__ == '__main__':
    main()



