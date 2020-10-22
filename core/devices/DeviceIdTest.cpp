// Device Id class test.
#include <gtest/gtest.h>
#include <ostream>
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/common/exceptions.h"

namespace {

using namespace arrus::devices;

// DeviceId

// DeviceId::parse

struct ParseCorrectParams {
    std::string idString;
    DeviceId expectedId;

    friend std::ostream &
    operator<<(std::ostream &os, const ParseCorrectParams &state) {
        os << "idString: " << state.idString << " expectedId: "
           << state.expectedId;
        return os;
    }
};

class DeviceIdCorrectParseTest
        : public testing::TestWithParam<ParseCorrectParams> {
};

TEST_P(DeviceIdCorrectParseTest, ParsesCorrectDeviceIds) {
    const DeviceId id = DeviceId::parse(GetParam().idString);
    EXPECT_EQ(GetParam().expectedId, id);
}

INSTANTIATE_TEST_CASE_P

(SimpleCases, DeviceIdCorrectParseTest,
 testing::Values(
         ParseCorrectParams{"Us4OEM:0", DeviceId(DeviceType::Us4OEM, 0)},
         ParseCorrectParams{"Us4OEM:1", DeviceId(DeviceType::Us4OEM, 1)},
         ParseCorrectParams{"ProbeAdapter:0",
                            DeviceId(DeviceType::ProbeAdapter, 0)},
         ParseCorrectParams{"Probe:3", DeviceId(DeviceType::Probe, 3)},
         ParseCorrectParams{"CPU:4", DeviceId(DeviceType::CPU, 4)},
         ParseCorrectParams{"GPU:0", DeviceId(DeviceType::GPU, 0)},
         ParseCorrectParams{"GPU:3", DeviceId(DeviceType::GPU, 3)}
 ));

INSTANTIATE_TEST_CASE_P

(TestComponentTrimming, DeviceIdCorrectParseTest,
 testing::Values(
         // Make sure that id components are trimmed before creating the id.
         // Any number of leading/trailing whitespaces is accepted.
         ParseCorrectParams{"Us4OEM : 0", DeviceId(DeviceType::Us4OEM, 0)},
         ParseCorrectParams{" Us4OEM :  1 ",
                            DeviceId(DeviceType::Us4OEM, 1)},
         ParseCorrectParams{" GPU:2   ", DeviceId(DeviceType::GPU, 2)}
 ));

struct ParseIncorrectParams {
    std::string idString;

    friend std::ostream &
    operator<<(std::ostream &os, const ParseIncorrectParams &state) {
        os << "idString: " << state.idString;
        return os;
    }
};

class DeviceIdIncorrectParseTest
        : public testing::TestWithParam<ParseIncorrectParams> {
};

TEST_P(DeviceIdIncorrectParseTest, DeclineIncorrectIds) {
    EXPECT_THROW(DeviceId::parse(GetParam().idString),
                 ::arrus::IllegalArgumentException);
}

INSTANTIATE_TEST_CASE_P

(TestUnknownDevice, DeviceIdIncorrectParseTest,
 testing::Values(
         ParseIncorrectParams{"Unknown: 0"},
         ParseIncorrectParams{"Unknown: 4"},
         ParseIncorrectParams{"Abcdefghijklmnopqsrtwxyz:4"}
 ));

INSTANTIATE_TEST_CASE_P

(TestInvalidCapitalization, DeviceIdIncorrectParseTest,
 testing::Values(
         // We require exact case name: Us4OEM
         ParseIncorrectParams{"US4OEM: 0"},
         ParseIncorrectParams{"gpu:1"}
));

INSTANTIATE_TEST_CASE_P

(TestNonAlphanumericCharacters, DeviceIdIncorrectParseTest,
 testing::Values(
         // We require exact case name: Us4OEM
         ParseIncorrectParams{"us_4oem:1"},
         ParseIncorrectParams{"GPU :1_2"},
         ParseIncorrectParams{"GPU :abc2"},
         ParseIncorrectParams{"GPU :2abc"},
         ParseIncorrectParams{"GPU :2 abc"}
 ));

INSTANTIATE_TEST_CASE_P

(TestMissingIdComponents, DeviceIdIncorrectParseTest,
 testing::Values(
         ParseIncorrectParams{""},
         ParseIncorrectParams{" : "},
         ParseIncorrectParams{"GPU:"},
         ParseIncorrectParams{" : 0"},
         ParseIncorrectParams{" GPU  0 "},
         ParseIncorrectParams{"GPU::0"}
 ));

INSTANTIATE_TEST_CASE_P

(TestInvalidOrdinalRange, DeviceIdIncorrectParseTest,
 testing::Values(
         ParseIncorrectParams{"GPU:-1"},
         ParseIncorrectParams{"GPU:1000000"}
 ));

// DeviceId::toString


// przetestowac, jak zachowa sie konwersja, gdy podamy DeviceType spoza zakresu enum

struct ToStringCorrectParams {
    DeviceId id;
    std::string expectedString;

    friend std::ostream &
    operator<<(std::ostream &os, const ToStringCorrectParams &state) {
        os <<  " id: " << state.id
           << "expected string: " << state.expectedString;
        return os;
    }
};

class DeviceIdToStringCorrectTest
        : public testing::TestWithParam<ToStringCorrectParams> {
};

TEST_P(DeviceIdToStringCorrectTest, ToStringCorrectIds) {
    EXPECT_EQ(GetParam().id.toString(), GetParam().expectedString);
}

INSTANTIATE_TEST_CASE_P

(SimpleCases, DeviceIdToStringCorrectTest,
 testing::Values(
         ToStringCorrectParams{DeviceId(DeviceType::Us4OEM, 0), "Us4OEM:0"},
         ToStringCorrectParams{DeviceId(DeviceType::GPU, 1), "GPU:1"}
 ));

}


