CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra"

.PHONY: all clean

all: nbody_0 nbody_a nbody_b nbody_c nbody_d
	@true

clean:
	rm -f *.o nbody_0 nbody_a nbody_b nbody_c nbody_d

%.o: %.cu utils.h
	nvcc $(CUFLAGS) $< -c -o $@

%: %.o main.o
	nvcc $(CUFLAGS) $^ -o $@
