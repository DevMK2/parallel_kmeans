SHELL = /bin/bash

CC = g++
CFLAGS = --std=c++14 -o 
INCLUDES = -Iinclude -I.
TEST_LINKS = -L./devel/lib -lgtest -lpthread -DDEBUG
vpath %.cc src tests

.SUFFIXES: .cc .o
.cc.o: 
	$(CC) -c $(CFLAGS) $@ $< $(TEST_LINKS) $(INCLUDES)

OBJS = evaluate.o flags.o log.o
TARGET = devel/lib/liblog.a
TEST_OBJS = evaluate_test.o flags_test.o time_lap_test.o log_test.o
TEST_TARGET = evaluate_test flags_test time_lap_test log_test

default_target: all
all: $(TARGET) clean
test: $(TARGET) $(TEST_TARGET) clean

$(TARGET): $(OBJS)
	ar rcs $@ $(OBJS)

$(TEST_TARGET): $(TEST_OBJS)
	mkdir -p temp &&\
	$(CC) $(CFLAGS) temp/$@ $@.o $(TARGET) $(TEST_LINKS) -llog &&\
	./temp/$@

clean:
	rm -rf temp && rm -rf *.o
