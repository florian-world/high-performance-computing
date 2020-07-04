#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import glob
import os

datafiles = sorted(glob.glob('output/plot2d_*.txt'))
print("Plotting {} frame(s).".format(len(datafiles)))
print("This will take about {}MB of space!".format(int(1.23 * len(datafiles))))
for filename in datafiles:
    print(filename)
    L = np.loadtxt(filename, skiprows=1)
    x = L[:, 0]
    y = L[:, 1]
    phi = L[:, 2]

    plt.clf()
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(0.0, 1.0);
    ax.set_ylim(0.0, 1.0);
    cmap = plt.cm.coolwarm

    # vmin and vmax is the range of phi values.
    plt.scatter(x, y, c=cmap(phi), vmin=0.0, vmax=1.0)
    plt.savefig(filename + '.raw')

print("Running ffmpeg to generate a video...")
try:
    os.system('ffmpeg -v panic -stats -framerate 20 -pix_fmt rgba -s 640x480 -i "output/plot2d_%03d.txt.raw" -c:v libx264 -r 20 -pix_fmt yuv420p -y "movie2d.mp4"')
except:
    print("================================================")
    print("Are you on Euler, did you forget to module load?")
    print("module load legacy ffmpeg/1.0.1")
    print("================================================")
    raise
print("Done!")
