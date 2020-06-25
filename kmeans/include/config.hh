#ifndef __CONFIG_HH__
#define __CONFIG_HH__

#include "npy.hh" // npy size_t
#include <string>

#define ORIGIN_K_SIZE 10
#define ORIGIN_DATA_SIZE 60000
#define ORIGIN_FEAT_SIZE 200

#define DATA_SCALE 1
#define THREASHOLD 20

static const size_t KSize = ORIGIN_K_SIZE;
static const size_t DataSize = ORIGIN_DATA_SIZE * DATA_SCALE;
static const size_t FeatSize = ORIGIN_FEAT_SIZE;

static int threashold = THREASHOLD;

static std::string LogFileName = "./results/kmeans_parallel_const_theashold10_scale1";
static std::string DefaultInputFile = "../mnist/mnist_encoded/encoded_train_ae.npy";

static const size_t MaxDataValue = 987654321.0;

#endif
