#include <gtest/gtest.h>
#include <ostream>
#include "arrus/core/common/logging.h"
#include "arrus/core/common/collections.h"
#include "arrus/common/logging/impl/Logging.h"
#include "Us4OEMSettingsValidator.h"

namespace {
using namespace arrus;

struct TestUs4OEMSettings {
    std::vector<ChannelIdx> channelMapping{getRange<ChannelIdx>(0, 128)};
    BitMask activeChannelGroups{getNTimes<bool>(true, 16)};
    std::optional<uint16> dtgcAttenuation{42};
    uint16 pgaGain{30};
    uint16 lnaGain{24};
    TGCCurve tgcSamples{getRange<float>(30, 40, 0.5)};
    uint32 lpfCutoff{(int) 10e6};
    std::optional<uint16> activeTermination{50};

    std::vector<std::string> invalidParameters;

    Us4OEMSettings getUs4OEMSettings() {
        return Us4OEMSettings(channelMapping, activeChannelGroups,
                              dtgcAttenuation, pgaGain,
                              lnaGain, lpfCutoff, activeTermination,
                              tgcSamples);
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TestUs4OEMSettings &settings) {
        os << "channelMapping: " << toString(settings.channelMapping)
           << " activeChannelGroups: " << toString(settings.activeChannelGroups)
           << " dtgcAttenuation: " << toString(settings.dtgcAttenuation)
           << " pgaGain: " << (int) settings.pgaGain
           << " lnaGain: " << (int) settings.lnaGain
           << " lpfCutoff: " << settings.lpfCutoff
           << " activeTermination: " << toString(settings.activeTermination)
           << " tgcSamples: " << toString(settings.tgcSamples);

        for(const auto& invalidParameter : settings.invalidParameters) {
           os << " invalidParameter: " << invalidParameter;
        }
        return os;
    }
};

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
         TestUs4OEMSettings{.channelMapping={
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
                 120, 121, 122, 123, 124, 125, 126, 127}},
         TestUs4OEMSettings{.activeChannelGroups={
                 false, false, false, true, true, true, true, true,
                 true, false, true, true, true, true, true, true}},
         TestUs4OEMSettings{.dtgcAttenuation=6},
         TestUs4OEMSettings{.dtgcAttenuation={}}, // Turn off
         TestUs4OEMSettings{.pgaGain=24},
         TestUs4OEMSettings{.lnaGain=24},
         TestUs4OEMSettings{.lpfCutoff=(int) 15e6},
         TestUs4OEMSettings{.activeTermination=200},
         TestUs4OEMSettings{.activeTermination={}}, // Turn off
         TestUs4OEMSettings{.tgcSamples={14.0f, 20.0f, 25.0f}},
         TestUs4OEMSettings{.tgcSamples={}} // Turn off
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
         TestUs4OEMSettings{
                 .channelMapping = {0, 1, 2, 3, 4, 5, 6, 7, 8},
                 .invalidParameters = {"channel mapping"}},
         // Invalid mapping (channel mapping are mixed between 32-element groups)
         TestUs4OEMSettings{.channelMapping={
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
                 77}},

         // Invalid number of active channel groups
         TestUs4OEMSettings{
                 .activeChannelGroups = getNTimes(false, 15),
                 .invalidParameters = {"active channel groups"}},
         // Empty array of active channel groups
         TestUs4OEMSettings{
                 .activeChannelGroups = {},
                 .invalidParameters = {"active channel groups"}},
         // Invalid value
         TestUs4OEMSettings{
                 .pgaGain = 777,
                 .invalidParameters = {"pga gain"}},
         // Invalid value
         TestUs4OEMSettings{
                 .lnaGain = 666,
                 .invalidParameters = {"lna gain"}},
         // Invalid value
         TestUs4OEMSettings{
                 .dtgcAttenuation = 123,
                 .invalidParameters = {"dtgc attenuation"}},
         // Invalid value
         TestUs4OEMSettings{
                 .lpfCutoff = 1,
                 .invalidParameters = {"lpf cutoff"}},
         // Invalid value
         TestUs4OEMSettings{
                 .activeTermination = 9999,
                 .invalidParameters = {"active termination"}},

         // Invalid number of TGC samples
         TestUs4OEMSettings{
                 .tgcSamples = getNTimes(40.0f, 1024),
                 .invalidParameters = {"tgc samples"}},
         // Invalid TGC samples values (1) below the range
         TestUs4OEMSettings{
                 .pgaGain = 30,
                 .lnaGain = 24,
                 .tgcSamples = {13.0f},
                 .invalidParameters = {"tgc samples"}},
         // Invalid TGC samples values (2) above the range
         TestUs4OEMSettings{
                .pgaGain = 30,
                .lnaGain = 24,
                .tgcSamples = {55.0f},
                .invalidParameters = {"tgc samples"}},
         // Invalid TGC samples values (3) below 0
         TestUs4OEMSettings{
                .pgaGain = 24,
                .lnaGain = 12,
                .tgcSamples = {-1.0f},
                .invalidParameters = {"tgc samples"}},
         // Invalid TGC samples values (3) multiple wrong values
         TestUs4OEMSettings{
                 .pgaGain = 24,
                 .lnaGain = 24,
                 .tgcSamples = {0, 15, 50.0f},
                 .invalidParameters = {"tgc samples"}},
         TestUs4OEMSettings{
                 .pgaGain = 11,
                 .lnaGain = 22,
                 .invalidParameters = {"pga gain", "lna gain"}}
));

// Test that multiple errors are signeld

// Main
int main(int argc, char **argv) {
    INIT_ARRUS_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
}





