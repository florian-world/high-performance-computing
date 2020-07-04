CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra -fopenmp"

.PHONY: all clean

all: overlap
	@true

clean:
	rm -f overlap

%: %.cu utils.h
	nvcc $(CUFLAGS) $< -o $@
