CFLAGS = -O3
LDFLAGS = -lblas
CC =$(shell which mpicxx)
 
.SECONDARY: 
OBJECTS = auxiliar.o

.PHONY: all
all: $(OBJECTS)

%.o: %.cpp
	$(CC)  -c $(CFLAGS) $<
 
.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti 

	
