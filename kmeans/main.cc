#ifdef PARALLEL
#include "kmeans_parallel.cuh"
#else
#include "kmeans.hh"
#endif
#include "datatype.hh"
#include "announce.hh"
#include "kmeans_argv_parse.hh"

int main(int argc, const char *argv[]) {
    DataPoint *data, *centroids;

    try {
        data = new DataPoint[DataSize];
        centroids = new DataPoint[KSize];
        std::string dataFile = parseFilePath(argc, argv);
        extractDataFromFile(dataFile, data);
    }
    catch(const std::bad_alloc& e) {
        fprintf(stderr, "%s\n", e.what()); return -1;
    }
    catch(const std::ifstream::failure& e) {
        fprintf(stderr, "%s\n", e.what()); return -1;
    }
    catch(const std::runtime_error& e) {
        fprintf(stderr, "%s\n", e.what()); return -1;
    }

    KMeans::initCentroids(centroids, data);
    announce.InitCentroids(centroids);
    KMeans::main(centroids, data);

    delete[] data;
    delete[] centroids;
    return 0;
}