#!/usr/bin/env python3

import matplotlib.image
import numpy as np
import glob
import subprocess

def main():
    files = glob.glob('output/dump-*.csv')
    for file in sorted(files):
        data = np.loadtxt(file, delimiter=',')
        png = file.replace('csv', 'png')
        matplotlib.image.imsave(png, data)
        print(png)

    mp4 = 'output/movie.mp4'
    print("Attempting to generate a movie '{}' with ffmpeg.".format(mp4))
    try:
        subprocess.check_call([
                'ffmpeg', '-stats', '-framerate', '10',
                '-i', 'output/dump-%*.png', '-c:v', 'libx264',
                '-r', '10', '-pix_fmt', 'yuv420p', '-y', 'output/movie.mp4'])
    except:
        print("Seems like ffmpeg is not available.")
    else:
        print("Movie saved to '{}'.".format(mp4))


if __name__ == '__main__':
    main()
