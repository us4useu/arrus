#include <gtest/gtest.h>
#include <ostream>

#include "arrus/common/format.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/probe/ProbeSettingsValidator.h"

namespace {
using namespace arrus;
using namespace arrus::devices;

struct TestProbeSettings {
    ProbeModelId modelId{"test", "test"};
    Tuple<ProbeModel::ElementIdxType> numberOfElements{192};
    Tuple<double> pitch{0.3e-3};
    Interval<float> txFrequencyRange = {1e6, 10e6};
    Interval<uint8> voltageRange = {0, 90};
    std::vector<ChannelIdx> channelMapping =
        arrus::getRange<ChannelIdx>(0, 192);

    [[nodiscard]] ProbeSettings toProbeSettings() const {
        return ProbeSettings{
            ProbeModel{modelId, numberOfElements, pitch, txFrequencyRange, voltageRange},
            channelMapping};
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TestProbeSettings &settings) {
        os << "modelId: " << settings.modelId
           << " numberOfElements: " << toString(settings.numberOfElements)
           << " pitch: " << toString(settings.pitch)
           << " txFrequencyRange: " << toString(settings.txFrequencyRange)
           << " channelMapping: " << toString(settings.channelMapping);
        return os;
    }
};

class CorrectProbeSettingsTest
    : public testing::TestWithParam<TestProbeSettings> {
};

TEST_P(CorrectProbeSettingsTest, AcceptsCorrect) {
    ProbeSettingsValidator validator(0);
    TestProbeSettings val = GetParam();
    validator.validate(val.toProbeSettings());
    EXPECT_NO_THROW(validator.throwOnErrors());
    validator.throwOnErrors();
}

INSTANTIATE_TEST_CASE_P

(ValidProbeSettings, CorrectProbeSettingsTest,
 testing::Values(
     // 1-D, all channels
     TestProbeSettings{},
     // 1-D, subset of the underlying adapter is used
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.numberOfElements = {96},
         x.channelMapping = arrus::concat(
            getRange<ChannelIdx>(0, 48),
            getRange<ChannelIdx>(144, 192)
         ))
     ),
     // 2-D, all channels are used
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.numberOfElements = {8, 8},
         x.pitch = {0.3e-3, 0.3e-3},
         x.channelMapping = getRange<ChannelIdx>(0, 64)
     )),
     // 2-D, some of the channels are used
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.numberOfElements = {16, 16},
         x.pitch = {0.3e-3, 0.3e-3},
         x.channelMapping = arrus::concat(
                 getRange<ChannelIdx>(0, 128),
                 getRange<ChannelIdx>(512, 640)
         )
     ))
 ));


class IncorrectProbeSettingsTest
    : public testing::TestWithParam<TestProbeSettings> {
};

TEST_P(IncorrectProbeSettingsTest, RejectsIncorrect) {
    ProbeSettingsValidator validator(0);
    TestProbeSettings val = GetParam();
    validator.validate(val.toProbeSettings());
    EXPECT_THROW(validator.throwOnErrors(), ::arrus::IllegalArgumentException);
}

INSTANTIATE_TEST_CASE_P

(InvalidProbeSettings, IncorrectProbeSettingsTest,
 testing::Values(
     // Testing probe model
     // - Negative pitch
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.pitch = {-.3e-3}
     )),
     // - Negative frequency
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.txFrequencyRange = {-10e6, 10e6}
     )),
     // - Invalid number of channels in mapping
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.numberOfElements = {96},
         x.channelMapping = getRange<ChannelIdx>(0, 192)
     )),
     // - Non-unique channels
     ARRUS_STRUCT_INIT_LIST(TestProbeSettings, (
         x.numberOfElements = {32},
         x.channelMapping = ::arrus::concat(
             getRange<ChannelIdx>(0, 2),
             getRange<ChannelIdx>(0, 30)
         )
     ))
 ));

}

