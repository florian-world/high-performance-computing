#!/usr/local/bin/python3
import matplotlib.pyplot as plt


fname = '../output.txt'

if __name__ == '__main__':
    print("Plotting results..")

    time = []
    s1   = []
    s2   = []
    with open(fname) as f:
        for line in f:
            t, a, b = line.split()
            time.append(float(t))
            s1.append(float(a))
            s2.append(float(b))


    plt.plot(time, s1, label='S1')
    plt.plot(time, s2, label='S2')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Sepcies Quantity')
    plt.show()
