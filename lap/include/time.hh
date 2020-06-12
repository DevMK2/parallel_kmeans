#ifndef __TIME_HH__
#define __TIME_HH__
#include <string>
#include <time.h>

using Clock = double;
inline std::string ClockToString(const Clock& clock) {
    return std::to_string(clock);
}

namespace Time {

static clock_t __lapTime;
static constexpr Clock clockPerSec = (Clock)CLOCKS_PER_SEC;
static constexpr Clock clockPerMSec = (Clock)CLOCKS_PER_SEC/1000;

inline void Reset() {
    __lapTime = clock();
}

inline clock_t __elpased() {
    clock_t prev = __lapTime;
    Time::Reset();
    return (__lapTime - prev);
}

inline Clock ElapsedSec() {
    return (Clock)__elpased()/clockPerSec;
}

inline Clock ElapsedMSec() {
    return (Clock)__elpased()/clockPerMSec;
}

inline Clock GetClock() {
    return (Clock)clock()/clockPerMSec;
}
};

#endif
