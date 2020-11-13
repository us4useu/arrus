#include <gtest/gtest.h>
#include <ostream>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/devices/utils.h"
#include "arrus/core/common/tests.h"

namespace {

struct GetPathRootTestCase {
    std::string path;
    std::pair<std::string, std::string> expectedRootTail;

    friend std::ostream &
    operator<<(std::ostream &os, const GetPathRootTestCase &aCase) {
        os << "path: " << aCase.path << " expectedRootTail: "
           << aCase.expectedRootTail.first
           << ", " << aCase.expectedRootTail.second;
        return os;
    }
};

class GetPathRootTest
        : public testing::TestWithParam<GetPathRootTestCase> {
};

TEST_P(GetPathRootTest, CorrectlyExtractsRootAndTail) {
    GetPathRootTestCase tc = GetParam();
    auto[root, tail] = ::arrus::devices::getPathRoot(tc.path);

    EXPECT_EQ(root, tc.expectedRootTail.first);
    EXPECT_EQ(tail, tc.expectedRootTail.second);
}

INSTANTIATE_TEST_CASE_P

(TestingCorrectGetPathRoot, GetPathRootTest,
 testing::Values(
         ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (
                 x.path = "/Us4R:0",
                 x.expectedRootTail = {"Us4R:0", ""}
         )),
         ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (
                 x.path = "/Us4R:0/Probe:0",
                 x.expectedRootTail = {"Us4R:0", "/Probe:0"}
         )),
         ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (
                 x.path = "/Us4R:0/Us4OEM:3",
                 x.expectedRootTail = {"Us4R:0", "/Us4OEM:3"}
         )),
         ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (
                 x.path = "/Us4R:0/Us4OEM:3/Sequencer:0",
                 x.expectedRootTail = {"Us4R:0", "/Us4OEM:3/Sequencer:0"}
         ))
 )
);

class GetPathRootInvalidInputTest
        : public testing::TestWithParam<GetPathRootTestCase> {
};

TEST_P(GetPathRootInvalidInputTest, GetPathRootInvalidInputTest) {
    GetPathRootTestCase tc = GetParam();
    EXPECT_THROW(::arrus::devices::getPathRoot(tc.path),
                 ::arrus::IllegalArgumentException);
}

INSTANTIATE_TEST_CASE_P
(InvalidDataTest, GetPathRootInvalidInputTest,
testing::Values(
        ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (x.path = "")),
        ARRUS_STRUCT_INIT_LIST(GetPathRootTestCase, (x.path = "/"))
));

}
