CXX=h5pcc
CXXFLAGS = -Wpedantic -Wall -Wextra -std=c++11 -lstdc++ -O3 -g 

all: main

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)
clean:
	rm -rf *.o *.xmf *h5 *txt main

main: main.o auxiliary.o wave.o
	$(CXX) $(CXXFLAGS)-I. -o main $^

run:
	make clean ; make ; mpirun -n 8 ./main 100 2 1.0	
run1:
	make clean ; make ; mpirun -n 1 ./main 100 1 1.0	

.PHONY: all clean run run1
