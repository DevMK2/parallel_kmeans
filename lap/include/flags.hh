#ifndef __FLAGS_HH__
#define __FLAGS_HH__

#include <unordered_map>

using uint8 = __uint8_t;
using FlagsID_T = std::unordered_map<std::string, uint8>; 

class Flags {
public:
    static constexpr uint8 MaxFlags = (uint8)(0-1);

private:
    FlagsID_T flags;

public:
    uint8 Size();
    uint8 GetID(const std::string& flag);

private:
    uint8 lastID = 0;
    uint8 uid();
};

#endif
