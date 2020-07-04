CXX = upcxx
CXXFLAGS += -O3 -std=c++17
DEPS = factor.a

all: sequential divideAndConquer producerConsumer

sequential: sequential.cpp factor.a
	$(CXX) $(CXXFLAGS) -o $@ $^

divideAndConquer: divideAndConquer.cpp factor.a
	$(CXX) $(CXXFLAGS) -o $@ $^

producerConsumer: producerConsumer.cpp factor.a
	$(CXX) $(CXXFLAGS) -o $@ $^


clean:; rm -rf *.o sequential divideAndConquer producerConsumer
