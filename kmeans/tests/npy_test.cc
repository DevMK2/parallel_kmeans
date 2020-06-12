/***********************************************/
/*                                             */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*                                             */
/***********************************************/
#include "npy.hh"
#include <fstream>

const int Nx = 128;
const int Ny = 64;
const int Nz = 32;

void readAndWriteTest();
void isSameToPythonTest();

int main() {
    readAndWriteTest();
    isSameToPythonTest();
}

void readAndWriteTest() {
    //set random seed so that result is reproducible (for testing)
    srand(0);
    //create random data
    std::vector<std::complex<double>> data(Nx*Ny*Nz);
    for(int i = 0;i < Nx*Ny*Nz;i++)
        data[i] = std::complex<double>(rand(),rand());
    
    // save data
    NPY::saveOverwrite<std::complex<double>>("array1.npy", data, {Nz,Ny,Nx});

    //load it into a new array
    auto arr = NPY::load<std::complex<double>>("array1.npy");
    {
    assert(arr.word_size == sizeof(std::complex<double>));
    assert(arr.shape.size() == 3 && arr.shape[0] == Nz && arr.shape[1] == Ny && arr.shape[2] == Nx);
    }
    auto loaded_data = NPY::extract<std::complex<double>>(arr);
    
    // confirm data
    for(int i = 0; i < Nx*Ny*Nz;i++)
        assert(data[i] == loaded_data[i]);

    //append the same data to file
    //npy array on file now has shape (Nz+Nz,Ny,Nx)
    NPY::saveAppend<std::complex<double>>("array1.npy", data, {Nz,Ny,Nx});
}

void isSameToPythonTest() {
    std::cout << "isSameToPython test " << std::endl;

    float maximumError = 0;

    std::ifstream in("../tests/mnist_seq.txt");
    if(in.is_open()) {
        auto arr = NPY::load<float>("../../mnist/mnist_encoded/encoded_train_ae.npy");
        auto data = NPY::extract<float>(arr);
        size_t size = arr.shape[1] * arr.shape[0];
        int i = 0;

        std::string s;

        while( in >> s ) {
            float error = std::atof(s.c_str())-data[i++];
            maximumError = std::max(maximumError, error);
            assert(error < 0.0001);
        }
    }
    else {
        std::cout << "No such file" << std::endl;
    }

    std::cout << "maximum deviation error :" << std::to_string(maximumError) << std::endl;
}
