#include <gtest/gtest.h>
#include "../src/log.cc"

std::vector<std::string> TockenizedResult(const std::string& result);
std::vector<std::string> TockenizedResultFromFile(const std::string& fileName);

TEST(Warterfall, ResultFileExists) {
    std::string path = "temp/ResultFileExists";
    {
        Log<> log(path);
        log.Lap("a");
        log.Lap("b");
        log.Lap("c");
    }

    std::ifstream check(path+".csv");
    ASSERT_TRUE(check.is_open());
}

TEST(Warterfall, ResultFileFormatCorrectJustBegin) {

    auto correctTokens = std::vector<std::string>{ "message", "elapsed" };

    std::string path = "temp/ResultFileFormatCorrectJustBegin";
    {
        Log<> log(path);
    }

    auto resultTokens = TockenizedResultFromFile(path);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());
    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}

TEST(Warterfall, ResultFileFormatCorrect) {
    std::string msg1 = "yaho11!!";
    std::string msg2 = "havesomewhitespace";
    std::string msg3 = "_";

    auto correctTokens = std::vector<std::string>{
        "message", "elapsed", msg1, "0", msg2, "0",  msg3, "0"
    };

    std::string path = "temp/ResultFileFormatCorrect";
    {
        Log<> log(path);
        for(int i=0; i!=50000; ++i);
        log.Lap(msg1);
        for(int i=0; i!=30000; ++i);
        log.Lap(msg2);
        for(int i=0; i!=10000; ++i);
        log.Lap(msg3);
    }

    auto resultTokens = TockenizedResultFromFile(path);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());
    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}

TEST(Loop, ResultFileExists) {
    std::string path = "temp/LoopResultFileExists";
    {
        Log<LoopEvaluate,1024> log(path);
        log.Lap("a");
        log.Lap("b");
        log.Lap("c");
    }

    std::ifstream check(path+".csv");
    ASSERT_TRUE(check.is_open());
}

TEST(Loop, LoopResultFileFormatCorrectJustBegin) {
    auto correctTokens = std::vector<std::string>{
        "message",     "iteration",    "first", "last", "average", "max", "min"
    };

    std::string path = "temp/LoopResultFileFormatCorrectJustBegin";
    {
    Log<LoopEvaluate,1024> log(path);
    }

    auto resultTokens = TockenizedResultFromFile(path);

    ASSERT_EQ(correctTokens.size(), resultTokens.size());
    for(int i=correctTokens.size()-1; i>=0; --i) {
        if(correctTokens[i] != "0")
            ASSERT_EQ(correctTokens[i], resultTokens[i]);
    }
}

TEST(Loop, LoopResultFileFormatCorrect) {
    int iter = 10;
    std::string iterStr = std::to_string(iter);
    std::string msg1 = "yaho11!!";
    std::string msg2 = "havesomewhitespace";
    std::string msg3 = "_";

    auto correctTokens = std::vector<std::string>{
        "message",     "iteration",    "first", "last", "average", "max", "min",
           msg1,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg2,         iterStr,        "0",     "0",     "0",     "0",   "0",
           msg3,         iterStr,        "0",     "0",     "0",     "0",   "0"
    };

    std::string path = "temp/LoopResultFileFormatCorrect";
    {
    Log<LoopEvaluate,1024> log(path);
    for(int loop=0; loop!=iter; ++loop) {
        for(int i=0; i!=50000; ++i);
        log.Lap(msg1);
        for(int i=0; i!=30000; ++i);
        log.Lap(msg2);
        for(int i=0; i!=10000; ++i);
        log.Lap(msg3);
    }
    }

    auto resultTokens = TockenizedResultFromFile(path);

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

std::vector<std::string> TockenizedResultFromFile(const std::string& fileName) {
    std::ifstream check(fileName+".csv");

    std::string result, line;
    while(check >> line)
        result+=line+"\n";

    return TockenizedResult(result);
}
