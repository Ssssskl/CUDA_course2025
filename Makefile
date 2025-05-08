NVCC       := nvcc

NVCCFLAGS  := -O3 -arch=sm_60 -std=c++11 -Xcompiler -fopenmp

TARGET     := jacob
SRC        := jacob.cu

.PHONY: all float double clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

float: NVCCFLAGS += -DUSE_FLOAT
float: clean all
double: clean all

clean:
	rm -f $(TARGET)