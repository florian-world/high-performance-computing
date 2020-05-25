#!/usr/bin/env python3

import matplotlib.pyplot as plt

fname = 'output.txt'

if __name__ == '__main__':
    print("Plotting results..")

    time = []
    sa   = []
    sb   = []
    with open(fname) as f:
        for line in f:
            t, a, b = line.split()
            time.append(float(t))
            sa.append(float(a))
            sb.append(float(b))


    plt.plot(time, sa, label='Sa')
    plt.plot(time, sb, label='Sb')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Species Quantity')
    plt.show()
