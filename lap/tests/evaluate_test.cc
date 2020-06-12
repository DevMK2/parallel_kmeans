#include <gtest/gtest.h>
#include "../src/evaluate.cc"
std::vector<std::string> TockenizedResult(const std::string& result);

TEST(Warterfall, yourLapWillBeEmptyAfter_Do) {
    TimeLap lap;
    for(int i=1; i!=127; ++i) //there is default BEGIN msg on front
        lap.PushAndCheckFull("a");

    Evaluate* eval = new WarterfallEvaluate;

    eval->Do(&lap);
    ASSERT_TRUE(lap.empty());

    delete eval;
}

TEST(Warterfall, ResultTextFormatTest) {
    std::string msg1 = "yaho11!!";
    std::string msg2 = "have some whitespace";
    std::string msg3 = "_";

    auto correctTokens = std::vector<std::string>{
        "message", "elapsed", msg1, "0", msg2, "0",  msg3, "0"
    };

    TimeLap lap;
    for(int i=0; i!=50000; ++i);
    lap.PushAndCheckFull(msg1);
    for(int i=0; i!=100000; ++i);
    lap.PushAndCheckFull(msg2);
    for(int i=0; i!=150000; ++i);
    lap.PushAndCheckFull(msg3);

    // using evaluate /////////////////////////
    Evaluate* eval = new WarterfallEvaluate;

    eval->Do(&lap);
    std::string result = eval->ToString();

    delete eval;
    ///////////////////////////////////////////

    auto resultTokens = TockenizedResult(result);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());

    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}

TEST(Loop, yourLapWillBeEmptyAfter_Do) {
    TimeLap lap;
    for(int i=1; i!=127; ++i) //there is default BEGIN msg on front
        lap.PushAndCheckFull("a");

    Evaluate* eval = new LoopEvaluate;

    eval->Do(&lap);
    ASSERT_TRUE(lap.empty());

    delete eval;
}

TEST(Loop, ResultTextFormatTest) {
    int iter = 10;
    std::string iterStr = std::to_string(iter);
    std::string msg1 = "yaho11!!";
    std::string msg2 = "have some whitespace";
    std::string msg3 = "_";

    auto correctTokens = std::vector<std::string>{
        "message",     "iteration",    "first", "last", "average", "max", "min",
           msg1,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg2,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg3,         iterStr,        "0",     "0",     "0",     "0",   "0"
    };

    TimeLap lap;
    for(int loop=0; loop!=iter; ++loop) {
        for(int i=0; i!=50000; ++i);
        lap.PushAndCheckFull(msg1);
        for(int i=0; i!=100000; ++i);
        lap.PushAndCheckFull(msg2);
        for(int i=0; i!=150000; ++i);
        lap.PushAndCheckFull(msg3);
    }

    // using evaluate /////////////////////////
    Evaluate* eval = new LoopEvaluate;

    eval->Do(&lap);
    std::string result = eval->ToString();

    delete eval;
    ///////////////////////////////////////////

    auto resultTokens = TockenizedResult(result);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());

    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}

TEST(Loop, ResultTextFormatTestInLargeLoop) {
    int iter = 10000; /*TODO Expand it !!*/
    std::string iterStr = std::to_string(iter);
    std::string msg1  = "yaho11!!";
    std::string msg2  = "have some whitespace";
    std::string msg3  = "_";
    std::string msg4  = "test 4";
    std::string msg5  = "test 5";
    std::string msg6  = "test 6";
    std::string msg7  = "test 7";
    std::string msg8  = "test 8";
    std::string msg9  = "test 9";
    std::string msg10 = "test10";

    auto correctTokens = std::vector<std::string>{
        "message",     "iteration",    "first", "last", "average", "max", "min",
           msg1,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg2,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg3,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg4,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg5,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg6,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg7,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg8,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg9,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg10,        iterStr,        "0",     "0",     "0",     "0",   "0"
    };

    TimeLap lap;
    for(int loop=0; loop!=iter; ++loop) {
        for(int i=0; i!=50; ++i);  lap.PushAndCheckFull(msg1);
        for(int i=0; i!=100; ++i); lap.PushAndCheckFull(msg2);
        for(int i=0; i!=150; ++i); lap.PushAndCheckFull(msg3);
        for(int i=0; i!=200; ++i); lap.PushAndCheckFull(msg4);
        for(int i=0; i!=250; ++i); lap.PushAndCheckFull(msg5);
        for(int i=0; i!=300; ++i); lap.PushAndCheckFull(msg6);
        for(int i=0; i!=350; ++i); lap.PushAndCheckFull(msg7);
        for(int i=0; i!=400; ++i); lap.PushAndCheckFull(msg8);
        for(int i=0; i!=450; ++i); lap.PushAndCheckFull(msg9);
        for(int i=0; i!=500; ++i); lap.PushAndCheckFull(msg10);
    }

    // using evaluate /////////////////////////
    Evaluate* eval = new LoopEvaluate;

    eval->Do(&lap);
    std::string result = eval->ToString();

    delete eval;
    ///////////////////////////////////////////

    auto resultTokens = TockenizedResult(result);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());

    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}


int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

std::vector<std::string> TockenizedResult(const std::string& result) {
    std::vector<std::string> ret;

    std::stringstream lineSS(result);
    std::string line;

    while(std::getline(lineSS, line, '\n')) {
        std::stringstream tokenSS(line);
        std::string token;

        while(std::getline(tokenSS, token, ',')) {
            ret.push_back(token);
        }
    }

    return ret;
}

