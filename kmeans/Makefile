SHELL = /bin/bash

CC = g++
CU = nvcc
CUFLAGS =#-lineinfo -arch=sm_50 -rdc=true ## for debug and profiling
CFLAGS = --std=c++14 -o
INCLUDES = -Iinclude -I./sequential/include -I./parallel/include
LIBDIR = -Llib
LIBS= -lcnpy -lz
CULIBS= -lcudart -lcudadevrt

.SUFFIXES: .cc .cu .o
vpath %.cc sequential/src
vpath %.cu parallel/src

.cc.o: 
	$(CC) -c $(CFLAGS) $@ $< $(INCLUDES)

.cu.o: 
	$(CU) $(CUFLAGS) -c $(CFLAGS) $@ $< $(INCLUDES)

default_target: all
all: kmeans_sequential kmeans_parallel

kmeans_sequential: main.o kmeans.o
	$(CC) -g $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) &&\
		./$@ ../mnist/mnist_encoded/encoded_train_ae.npy

kmeans_parallel: main.o kmeans_parallel.o
	cp main.cc main.cu &&\
	$(CU) $(CUFLAGS) $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) $(CULIBS) -DPARALLEL &&\
		./$@ ../mnist/mnist_encoded/encoded_train_ae.npy &&\
		rm -rf main.cu

clean:
	rm -rf *.o &&\
		rm -rf *.csv &&\
		rm -rf main.cu &&\
		rm -rf kmeans_sequential &&\
		rm -rf kmeans_parallel