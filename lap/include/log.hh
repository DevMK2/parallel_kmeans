#ifndef __LOG_HH__
#define __LOG_HH__

#include "time.hh"
#include "write_file.hh"
#include "time_lap.hh"
#include "evaluate.hh"

#include <memory>
using LapPtr = std::unique_ptr<TimeLap>;
using EvaluatePtr = std::unique_ptr<Evaluate>;
using WriteFilePtr = std::unique_ptr<WriteFile>;

template <class EvaluateType=WarterfallEvaluate, unsigned int DumpSize=128>
class Log {
protected:
    LapPtr lap;
    EvaluatePtr evaluate;
    std::string fileName;
    //WriteFilePtr writeFile;

public:
    Log(const std::string& file);
    ~Log();
    void Begin();
    void Lap(const std::string& msg);

private:
    void dump();
};

#endif
