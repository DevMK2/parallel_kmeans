#ifndef __EVALUATE_HH__
#define __EVALUATE_HH__

#include <string>
#include <vector>
#include "time.hh"
#include "time_lap.hh"
#include "flags.hh"

class Evaluate {
public:
    virtual void Do(TimeLap* lap) = 0;
    virtual const std::string ToString() = 0;
};

//////////////////////////////////////////////////////////////
struct WarterfallData {
    std::string message;
    Clock elapsedTime;
};

class WarterfallEvaluate : public Evaluate {
private:
    std::vector<WarterfallData> datas;

public:
    virtual void Do(TimeLap* lap) override;
    virtual const std::string ToString() override;
};

//////////////////////////////////////////////////////////////
struct LoopData {
    std::string message;
    unsigned int iteration;
    Clock last;
    Clock accum;
    Clock max;
    Clock min;
    Clock first;
};

class LoopEvaluate : public Evaluate {
private:
    Flags flags; // interpret from message to integer flags(index of datas)
    std::vector<LoopData> datas;

public:
    virtual void Do(TimeLap* lap) override;
    virtual const std::string ToString() override;
};

#endif
