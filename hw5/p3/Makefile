CUFLAGS=-O3 -std=c++14 --compiler-options "-Wall -Wextra"

.PHONY: all clean

all: electrostatics
	@true

clean:
	rm -rf electrostatics output

%: %.cu utils.h
	nvcc $(CUFLAGS) $< -o $@
