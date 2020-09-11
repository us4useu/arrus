#include <gtest/gtest.h>
#include <ostream>
#include "arrus/core/common/tests.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/logging/impl/Logging.h"
#include "Us4OEMSettingsValidator.h"

namespace {
using namespace arrus;
using namespace arrus::devices;

#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"


class CorrectUs4OEMSettingsTest
        : public testing::TestWithParam<TestUs4OEMSettings> {
};

TEST_P(CorrectUs4OEMSettingsTest, ValidateCorrectUs4OEMSettings) {
    Us4OEMSettingsValidator validator(0);
    TestUs4OEMSettings val = GetParam();
    validator.validate(val.getUs4OEMSettings());
    EXPECT_NO_THROW(validator.throwOnErrors());
}

INSTANTIATE_TEST_CASE_P

(ValidUs4OEMSettings, CorrectUs4OEMSettingsTest,
 testing::Values(
         TestUs4OEMSettings{},
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.channelMapping={
                 // 0-31
                 26, 27, 25, 23, 28, 22, 20, 21,
                 24, 18, 19, 15, 17, 16, 29, 13,
                 11, 14, 30, 8, 12, 5, 10, 9,
                 31, 7, 3, 6, 0, 2, 4, 1,
                 // 32-63
                 56, 55, 54, 53, 57, 52, 51, 49,
                 50, 48, 47, 46, 44, 45, 58, 42,
                 43, 59, 40, 41, 60, 38, 61, 39,
                 62, 34, 37, 63, 36, 35, 32, 33,
                 // 64-95
                 65, 67, 66, 69, 64, 68, 71, 70,
                 72, 74, 73, 75, 76, 77, 78, 79,
                 80, 82, 81, 83, 85, 84, 87, 86,
                 88, 92, 89, 94, 90, 91, 95, 93,
                 // 96-127
                 96, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119,
                 120, 121, 122, 123, 124, 125, 126, 127})),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.activeChannelGroups={
                 false, false, false, true, true, true, true, true,
                 true, false, true, true, true, true, true, true})),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.dtgcAttenuation=6)),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.dtgcAttenuation={})), // Turn off
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.pgaGain=24)),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.lnaGain=24)),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.lpfCutoff=(int) 15e6)),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.activeTermination=200)),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.activeTermination={})), // Turn off
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.tgcSamples={14.0f, 20.0f, 25.0f})),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.tgcSamples={})) // Turn off
 ));

class InCorrectUs4OEMSettingsTest
        : public testing::TestWithParam<TestUs4OEMSettings> {
};

TEST_P(InCorrectUs4OEMSettingsTest, ValidateInCorrectUs4OEMSettings) {
    Us4OEMSettingsValidator validator(0);
    TestUs4OEMSettings val = GetParam();
    validator.validate(val.getUs4OEMSettings());
    EXPECT_THROW(validator.throwOnErrors(), IllegalArgumentException);
    for(const auto& invalidParameter : GetParam().invalidParameters) {
        EXPECT_FALSE(validator.getErrors(invalidParameter).empty());
    }
}

INSTANTIATE_TEST_CASE_P

(InvalidUs4OEMSettings, InCorrectUs4OEMSettingsTest,
 testing::Values(
         // Invalid size of the channel mapping
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.channelMapping = {0, 1, 2, 3, 4, 5, 6, 7, 8},
                 x.invalidParameters = {"channel mapping"})),
         // Invalid mapping (channel mapping are mixed between 32-element groups)
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (x.channelMapping={
                 // 0-31, missing 11
                 26, 27, 25, 23, 28, 22, 20, 21,
                 24, 18, 19, 15, 17, 16, 29, 13,
                 14, 30, 8, 12, 5, 10, 9,
                 31, 7, 3, 6, 0, 2, 4, 1,
                 42,
                 // 32-63, missing 42
                 56, 55, 54, 53, 57, 52, 51, 49,
                 50, 48, 47, 46, 44, 45, 58,
                 43, 59, 40, 41, 60, 38, 61, 39,
                 62, 34, 37, 63, 36, 35, 32, 33,
                 11,
                 // 64-95, missing 77
                 65, 67, 66, 69, 64, 68, 71, 70,
                 72, 74, 73, 75, 76, 78, 79,
                 80, 82, 81, 83, 85, 84, 87, 86,
                 88, 92, 89, 94, 90, 91, 95, 93,
                 123,
                 // 96-127, missing 123
                 96, 97, 98, 99, 100, 101, 102, 103,
                 104, 105, 106, 107, 108, 109, 110, 111,
                 112, 113, 114, 115, 116, 117, 118, 119,
                 120, 121, 122, 124, 125, 126, 127,
                 77})),

         // Invalid number of active channel groups
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.activeChannelGroups = getNTimes(false, 15),
                 x.invalidParameters = {"active channel groups"})),
         // Empty array of active channel groups
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.activeChannelGroups = {},
                 x.invalidParameters = {"active channel groups"})),
         // Invalid value
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.pgaGain = 777,
                 x.invalidParameters = {"pga gain"})),
         // Invalid value
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.lnaGain = 666,
                 x.invalidParameters = {"lna gain"})),
         // Invalid value
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.dtgcAttenuation = 123,
                 x.invalidParameters = {"dtgc attenuation"})),
         // Invalid value
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.lpfCutoff = 1,
                 x.invalidParameters = {"lpf cutoff"})),
         // Invalid value
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.activeTermination = 9999,
                 x.invalidParameters = {"active termination"})),

         // Invalid number of TGC samples
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.tgcSamples = getNTimes(40.0f, 1024),
                 x.invalidParameters = {"tgc samples"})),
         // Invalid TGC samples values (1) below the range
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.pgaGain = 30,
                 x.lnaGain = 24,
                 x.tgcSamples = {13.0f},
                 x.invalidParameters = {"tgc samples"})),
         // Invalid TGC samples values (2) above the range
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                x.pgaGain = 30,
                x.lnaGain = 24,
                x.tgcSamples = {55.0f},
                x.invalidParameters = {"tgc samples"})),
         // Invalid TGC samples values (3) below 0
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                x.pgaGain = 24,
                x.lnaGain = 12,
                x.tgcSamples = {-1.0f},
                x.invalidParameters = {"tgc samples"})),
         // Invalid TGC samples values (3) multiple wrong values
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.pgaGain = 24,
                 x.lnaGain = 24,
                 x.tgcSamples = {0, 15, 50.0f},
                 x.invalidParameters = {"tgc samples"})),
         ARRUS_STRUCT_INIT_LIST(TestUs4OEMSettings, (
                 x.pgaGain = 11,
                 x.lnaGain = 22,
                 x.invalidParameters = {"pga gain", "lna gain"}))
));

// Test that multiple errors are signaled

// Main
int main(int argc, char **argv) {
    INIT_ARRUS_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
}





