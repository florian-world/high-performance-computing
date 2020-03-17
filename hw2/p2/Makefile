CFLAGS = -O3
LDFLAGS = -lblas
CC =$(shell which mpicxx)

.SECONDARY: 
BINARIES =  cannon

.PHONY: all auxiliar

all: auxiliar $(BINARIES)

auxiliar: 
	$(MAKE) -C auxiliar

cannon:  cannon.o auxiliar/auxiliar.o
	$(CC) -o $@ $^ $(LDFLAGS) 
	
%.o: %.cpp
	$(CC)  -c $(CFLAGS) $<
 
.PHONY: clean
clean:
	$(RM) $(BINARIES) auxiliar/*.o *.o *.ti 
	$(MAKE) -C auxiliar clean

	
