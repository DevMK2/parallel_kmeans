SHELL = /bin/bash

CC = g++
CU = nvcc
CUFLAGS =#-lineinfo -arch=sm_50 -rdc=true
CFLAGS = --std=c++14 -o
INCLUDES = -Iinclude -I./sequential/include -I./parallel/include -I../lap/src
LIBDIR = -Llib -L../lap/devel/lib
LIBS= -lcnpy -lz -llog
CULIBS= -lcudart -lcudadevrt

.SUFFIXES: .cc .cu .o
vpath %.cc sequential/src tests
vpath %.cu parallel/src tests

.cc.o: 
	$(CC) -c $(CFLAGS) $@ $< $(INCLUDES)

.cu.o: 
	$(CU) $(CUFLAGS) -c $(CFLAGS) $@ $< $(INCLUDES)

default_target: kmeans_sequential
#all: kmeans_sequential kmeans_parallel
npy_test_clean: npy_test clean

kmeans_sequential: main.o kmeans.o
	$(CC) -g $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) &&\
		./$@ ../mnist/mnist_encoded/encoded_train_ae.npy

kmeans_parallel: main.o kmeans_parallel.o
	cp main.cc main.cu &&\
	$(CU) $(CUFLAGS) -g $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) $(CULIBS) -DPARALLEL &&\
		nvprof ./$@ ../mnist/mnist_encoded/encoded_train_ae.npy &&\
		rm -rf main.cu

npy_test: npy_test.o
	python3 tests/make_mnist_seq.py &&\
	mkdir -p temp &&\
		$(CC) $(CFLAGS) temp/$@ $< $(LIBDIR) $(LIBS) &&\
		cd temp && ./$@ &&\
		cd .. && rm -rf temp

clean:
	rm -rf *.o &&\
		rm -rf *.csv &&\
		rm -rf main.cu &&\
		rm -rf kmeans_sequential &&\
		rm -rf kmeans_parallel
