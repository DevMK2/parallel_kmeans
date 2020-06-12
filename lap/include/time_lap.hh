#ifndef __LAP_HH__
#define __LAP_HH__

#include <queue>
#include "time.hh"

struct LapData {
    std::string message;
    Clock timeOccurrence;
};

class TimeLap : public std::queue<LapData>{
private:
    static constexpr const char* BEGIN = "__BEGIN";
    unsigned int fullSize = 128;
    unsigned int size = 0;
    bool popedBegin = false;

public:
    TimeLap() {
        this->Begin();
    }

    TimeLap(const unsigned int& fullSize) : fullSize(fullSize) {
        this->Begin();
    }

    inline void Begin() {
        this->size++;
        this->push(LapData{this->BEGIN, Time::GetClock()});
    }

    inline bool JustPopedBegin() {
        return this->popedBegin;
    }

    bool PushAndCheckFull(const std::string& msg)  {
        this->push(LapData{msg, Time::GetClock()});
        return fullSize <= ++this->size;
    }

    LapData PopAndCheckBegin() {
        LapData ret = this->front();
        this->pop();
        this->popedBegin = (ret.message == this->BEGIN);
        return ret;
    }
};

#endif
