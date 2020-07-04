CXX=h5pcc
CXXFLAGS = -Wpedantic -Wall -Wextra -std=c++11 -lstdc++ -g -O3 -fopenmp 

all: main
%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)
clean:
	rm -rf *.o *.xmf *h5 *txt main
main: main.o auxiliary.o wave.o
	$(CXX) $(CXXFLAGS)-I. -o main $^

run_mpi:
	make clean ; make ; mpirun -n 64 ./main 512 4 0.25

run_hybrid_v1:
	export OMP_NUM_THREADS=8 ; make clean ; make ; mpirun -n 8 --map-by ppr:3:node ./main 512 2 0.25

run_hybrid_v2:
	export OMP_NUM_THREADS=8 ; make clean ; make ; mpirun -n 8 --map-by ppr:2:node ./main 512 2 0.25

develop:
	export OMP_NUM_THREADS=8 ; make clean ; make ; mpirun -n 8 ./main 256 2 0.1


.PHONY: all clean run_mpi run_hybrid_v1 run_hybrid_v2 develop
