#include <gtest/gtest.h>
#include "../include/flags.hh"

TEST(FlagCanMakeOver255sID, make255Flags) {
    Flags flags;
    for(int i=0; i!=255; ++i)
        ASSERT_EQ(flags.GetID("My ID is "+std::to_string(i)), i);
}

TEST(FlagCanMakeOver255sID, 256thFlagThrowException) {
    Flags flags;
    for(int i=0; i!=255; ++i)
        ASSERT_EQ(flags.GetID("My ID is "+std::to_string(i)), i);

    EXPECT_THROW(flags.GetID("My ID is 255"), std::overflow_error);
}

TEST(FlagSize, FlagSizeAlwaysPlusOneToLastID) {
    Flags flags;
    for(int i=0; i!=255; ++i) {
        uint8 lastID = flags.GetID("My ID is "+std::to_string(i));
        ASSERT_EQ(flags.Size(), lastID+1);
    }
}

TEST(MaximumFlags, MaxFlagsEqual255) {
    ASSERT_EQ(Flags::MaxFlags, 255);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

