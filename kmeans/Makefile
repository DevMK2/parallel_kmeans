SHELL = /bin/bash

CC = g++
CU = nvcc
CFLAGS = --std=c++14 -o
#TODO lap build 추가
#TODO lap path 수정
INCLUDES = -Iinclude -I./sequential/include -I./parallel/include -I../lap/src
LIBDIR = -Llib -L../lap/devel/lib
#TODO libcudart 받아오기
LIBS= -lcnpy -lz -llog

.SUFFIXES: .cc .cu .o
vpath %.cc sequential/src tests
vpath %.cu parallel/src tests

.cc.o: 
	$(CC) -c $(CFLAGS) $@ $< $(INCLUDES)

.cu.o: 
	$(CU) -c $(CFLAGS) $@ $< $(INCLUDES)

default_target: kmeans_sequential
npy_test_clean: npy_test clean

kmeans_sequential: main_sequential.o kmeans.o
	$(CC) -g $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) &&\
		./$@ ../mnist/mnist_encoded/encoded_train_ae.npy

kmeans_sequential_sorting: main_sequential.o kmeans_sorting.o
	$(CC) -g $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) &&\
		./$@ ../mnist/mnist_encoded/encoded_train_ae.npy

kmeans_parallel: main_parallel.o kmeans_parallel.o
	$(CU) -g -G $(CFLAGS) $@ $? $(LIBDIR) $(LIBS) -lcudart &&\
		nvprof ./$@ ../mnist/mnist_encoded/encoded_train_ae.npy

npy_test: npy_test.o
	python3 tests/make_mnist_seq.py &&\
	mkdir -p temp &&\
		$(CC) $(CFLAGS) temp/$@ $< $(LIBDIR) $(LIBS) &&\
		cd temp && ./$@ &&\
		cd .. && rm -rf temp

clean:
	rm -rf *.o &&\
		rm -rf *.csv &&\
		rm -rf kmeans_sequential &&\
		rm -rf kmeans_sequential_sorting &&\
		rm -rf kmeans_parallel
