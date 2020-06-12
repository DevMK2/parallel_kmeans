#include "../include/evaluate.hh"

//Warterfall Evaluate/////////////////////////////////////////
void WarterfallEvaluate::Do(TimeLap* lap) {
    Clock prevClock = 0;

    while(!lap->empty()) {
        LapData data = lap->PopAndCheckBegin();

        if(lap->JustPopedBegin()) {
            prevClock = data.timeOccurrence;
            continue;
        }

        this->datas.emplace_back(
            WarterfallData {
                data.message,
                data.timeOccurrence - prevClock
            }
        );
        prevClock = data.timeOccurrence;
    }
}

const std::string WarterfallEvaluate::ToString() {
    std::string ret = "message,elapsed\n";
    for(auto data : this->datas) {
        ret += data.message;
        ret += "," + ClockToString(data.elapsedTime);
        ret += "\n";
    }
    return ret;
}

//Loop Evaluate///////////////////////////////////////////////
void LoopEvaluate::Do(TimeLap* lap) {
    Clock prevClock = 0;

    while(!lap->empty()) {
        LapData data = lap->PopAndCheckBegin();

        if(lap->JustPopedBegin()) {
            prevClock = data.timeOccurrence;
            continue;
        }

        Clock currElapsed = data.timeOccurrence - prevClock;

        int id = this->flags.GetID(data.message);
        if(this->datas.size() > id) {
            LoopData& currData = this->datas[id];
            currData.iteration += 1;
            currData.last = currElapsed;
            currData.accum += currElapsed;
            currData.max = std::max(currData.max, currElapsed);
            currData.min = std::min(currData.min, currElapsed);
        }
        else {
            this->datas.emplace_back(
                LoopData {
                    data.message, 1,currElapsed, currElapsed, currElapsed, currElapsed, currElapsed
                }
            );
        }

        prevClock = data.timeOccurrence;
    }
};

const std::string LoopEvaluate::ToString() {
    std::string ret = "message,iteration,first,last,average,max,min\n";
    for(auto data : this->datas) {
        ret += data.message;
        ret += "," + std::to_string(data.iteration);
        ret += "," + ClockToString(data.first);
        ret += "," + ClockToString(data.last);
        ret += "," + ClockToString(data.accum/data.iteration);
        ret += "," + ClockToString(data.max);
        ret += "," + ClockToString(data.min);
        ret += "\n";
    }
    return ret;
};
