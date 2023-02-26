CC=nvcc
TARGET=filter
SRC=src
INCLUDE=include
CFLAGS=-Xcompiler -Wall -g
OFLAGS=-O3 --m64 -use_fast_math
LFLAGS=-lfreeimage -lm
LLIBS=-I${HOME}/softs/FreeImage/include -L${HOME}/softs/FreeImage/lib/ -Iinclude

all: $(TARGET)

$(TARGET): $(SRC)/main.cu $(SRC)/kernel.cu # $(SRC)/utils.cu
	nvcc $(LLIBS) $(CFLAGS) $(OFLAGS) $(LFLAGS) $^ -o $@

main.cu: kernel.cu # utils.cu

kernel.cu : $(INCLUDE)/kernel.h # utils.cu

# utils.cu : $(INCLUDE)/utils.h 

clean:
	rm -f *.o $(TARGET)
