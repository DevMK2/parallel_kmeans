#include "../include/log.hh"

template <class EvaluateType, unsigned int DumpSize>
Log<EvaluateType, DumpSize>::Log(const std::string& file) : fileName(file+".csv") {
    this->lap.reset(new TimeLap(DumpSize));
    this->evaluate.reset(new EvaluateType);
}

template <class EvaluateType, unsigned int DumpSize>
Log<EvaluateType, DumpSize>::~Log() {
    this->dump();

    WriteFile::MakePtr(this->fileName)->Write(this->evaluate->ToString());
}

template <class EvaluateType, unsigned int DumpSize>
void Log<EvaluateType, DumpSize>::Begin() {
    this->lap->Begin();
}

template <class EvaluateType, unsigned int DumpSize>
void Log<EvaluateType, DumpSize>::Lap(const std::string& msg) {
    if(this->lap->PushAndCheckFull(msg)) {
        this->dump();
    }
}

template <class EvaluateType, unsigned int DumpSize>
void Log<EvaluateType, DumpSize>::dump() {
    this->evaluate->Do(this->lap.get());

    this->Begin();
}
