#ifndef __WRITE_FILE_HH__
#define __WRITE_FILE_HH__

#include <memory>
#include <fstream>
#include <iostream>
#include <cassert>

class WriteFile {
private:
    std::ofstream file;

public:
    WriteFile(const std::string& filePath) : file(filePath) {
        if(this->file.fail())
            std::cout << "Could't open the file : " << filePath << std::endl;
    }

    ~WriteFile() {
        if(this->file.is_open())
            this->file.close();
    }

    inline void Write(const std::string& texts) {
        assert(this->file.is_open());
        this->file << texts;
    }

    inline void Close() {
        this->file.close();
    }

    static std::unique_ptr<WriteFile> MakePtr(const std::string& filePath) {
        return std::move( std::make_unique<WriteFile>(filePath) );
    }
};

#endif
