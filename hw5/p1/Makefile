CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra -fopenmp"

.PHONY: all clean

all: benchmarks_a benchmarks_b benchmarks_c
	@true

clean:
	rm -f benchmarks_a benchmarks_b benchmarks_c

%: %.cu utils.h
	nvcc $(CUFLAGS) $< -o $@
