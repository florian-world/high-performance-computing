Compilation
===========

From the skeleton/p2/ folder, run

    mkdir -p build
    cd build
    cmake ..
    make -j8
    
Running
=======

From the build/ folder, run

    ./ssa

The code will generate a file called output.txt

Visualization
=============

From the build/ folder, run

    ../python/plot_output.py

You will need the `-Y` flag when connecting with ssh.
