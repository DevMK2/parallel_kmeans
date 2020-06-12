#include <gtest/gtest.h>
#include "../include/time_lap.hh"

TEST(CheckIsFull, defaultSize128) {
    TimeLap lap;

    //there is default BEGIN msg on front
    for(int i=1; i!=127; ++i) {
        ASSERT_FALSE(lap.PushAndCheckFull("a"));
    }
    // 128th Push 
    ASSERT_TRUE(lap.PushAndCheckFull("a"));
}

TEST(CheckIsFull, templateSize256) {
    TimeLap* lap = new TimeLap(256);

    //there is default BEGIN msg on front
    for(int i=1; i!=255; ++i) {
        ASSERT_FALSE(lap->PushAndCheckFull("a"));
    }
    // 256th Push 
    ASSERT_TRUE(lap->PushAndCheckFull("a"));

    delete lap;
}

TEST(CheckPopedBegin, firstMustPop_BEGIN) {
    TimeLap lap;

    lap.PopAndCheckBegin();
    ASSERT_TRUE(lap.JustPopedBegin());
}

TEST(CheckPopedBegin, ifPoped_BEGIN_JustPopedBegin_IsTrue) {
    TimeLap lap;

    lap.Begin();
    lap.Begin();
    lap.Begin();

    while(!lap.empty()) {
        lap.PopAndCheckBegin();
        ASSERT_TRUE(lap.JustPopedBegin());
    }
}

TEST(CheckPopedBegin, canCheck_BEGIN_betweenOtherMessages) {
    TimeLap lap;

    // even:=BEGIN, odd:=other message 
    for(int i=1; i!=128; ++i) {
        if(i%2 == 0)
            lap.Begin();
        else
            lap.PushAndCheckFull("a");
    }

    // even:=BEGIN, odd:=other message 
    int i = 0;
    while(!lap.empty()) {
        lap.PopAndCheckBegin();

        if(i%2 == 0)
            ASSERT_TRUE(lap.JustPopedBegin());
        else
            ASSERT_FALSE(lap.JustPopedBegin());
        ++i;
    }

    ASSERT_EQ(i, 128);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
