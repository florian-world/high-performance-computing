CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra -fopenmp"

.PHONY: all clean

all: reduction_sum reduction_argmax
	@true

clean:
	rm -f reduction_sum reduction_argmax

reduction_sum: reduction_sum.cu utils.h reduction_sum.h
	nvcc $(CUFLAGS) $< -o $@

reduction_argmax: reduction_argmax.cu utils.h reduction_argmax.h
	nvcc $(CUFLAGS) $< -o $@
