#ifndef __CONFIG_HH__
#define __CONFIG_HH__

#define DATA_SCALE 1
#define THREASHOLD 50 // FIXME

#include "npy.hh" // npy size_t
#include <string>

#define ORIGIN_K_SIZE 10
#define ORIGIN_DATA_SIZE 60000
#define ORIGIN_FEAT_SIZE 200

static const size_t KSize = ORIGIN_K_SIZE;
static const size_t DataSize = ORIGIN_DATA_SIZE * DATA_SCALE;
static const size_t FeatSize = ORIGIN_FEAT_SIZE;

static int threashold = THREASHOLD;

static std::string LogFileName = "";
static std::string DefaultInputFile = "../mnist/mnist_encoded/encoded_train_ae.npy";

static const size_t MaxDataValue = 987654321.0;

#endif
