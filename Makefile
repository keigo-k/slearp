CC = /usr/bin/gcc
CFLAGS = -O2 -Wall -lpthread
CXX = /usr/bin/g++
CXXFLAGS = -O2 -Wall -lpthread 
RM := rm -f

PROGRAM = slearp
OBJS = hashFea.o hashPronunceRule.o parseOption.o sequenceData.o discriminativeModel.o crfce.o sarow.o ssvm.o QPsolver.o ssmcw.o structurePredict.o 
HED = OptimizePara.h

$(PROGRAM): $(OBJS) $(HED)
	$(CXX) $(CXXFLAGS) $^  -o $@

.PHONY: clean

clean:
	$(RM) $(OBJS) $(PROGRAM)


