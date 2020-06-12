#include "../include/flags.hh"

constexpr uint8 Flags::MaxFlags;

uint8 Flags::Size() {
    return this->lastID;
}

uint8 Flags::GetID(const std::string& flag) {
    if(this->flags.find(flag) == this->flags.end())
        this->flags.emplace(flag, this->uid());

    return this->flags[flag];
}

uint8 Flags::uid() { // flag 삭제는 없으므로 단조증가만 해도 unique id가 됨.
    if(this->lastID == 255) {
        throw std::overflow_error("You only can make under 255 flags");
    }
    return this->lastID++; 
}
