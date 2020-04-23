CC=g++
LD=$(CC)

KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

CFLAGS = -Wall -Wfatal-errors -std=c++14 -I ./includes/
CFLAGS += -O3
CFLAGS += -fopenmp -D _OPENMP

OBJECTS = main_ssa.o SSA_CPU.o

.DEFAULT: all

all: ssa ssa_test cmaes

ssa: $(OBJECTS) SSA_CPU.cpp ./includes/SSA_CPU.hpp
	$(CC) $(CFLAGS) -c SSA_CPU.cpp -o SSA_CPU.o
	$(LD) $(OBJECTS) -o ssa $(CFLAGS)

ssa_test: SSA_CPU.o main_ssa_test.o SSA_CPU.cpp ./includes/SSA_CPU.hpp
	$(CC) $(CFLAGS) -c SSA_CPU.cpp -o SSA_CPU.o
	$(LD) SSA_CPU.o main_ssa_test.o -o ssa_test $(CFLAGS)

cmaes: $(OBJECTS) main_cmaes.cpp ./includes/objective.hpp 
	$(CC) -c main_cmaes.cpp -o main_cmaes.o $(KORALICFLAGS) $(KORALILIBS) $(CFLAGS)
	$(LD) main_cmaes.o SSA_CPU.o -o cmaes $(KORALICFLAGS) $(KORALILIBS) $(CFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $^ -o $@
	
clean:
	rm -f *.o
	rm -f ssa
	rm -f cmaes


