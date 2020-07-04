CXX=mpic++
CXXFLAGS = -Wpedantic -Wall -Wextra -std=c++11 -lstdc++ -O3 -g 

all: particle

%.o: %.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)
clean:
	rm -rf *.o particle

particle: particle.o
	$(CXX) $(CXXFLAGS)-I. -o particle $^


.PHONY: all clean
