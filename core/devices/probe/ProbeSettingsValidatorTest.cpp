#include <gtest/gtest.h>
#include <ostream>

#include "arrus/core/common/format.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/probe/ProbeSettingsValidator.h"

namespace {
using namespace arrus;

struct TestProbeSettings {
    ProbeModelId modelId{"test", "test"};
    Tuple<ProbeModel::ElementIdxType> numberOfElements{192};
    Tuple<double> pitch{0.3e-3};
    Interval<double> txFrequencyRange{1e6, 10e6};
    std::vector<ChannelIdx> channelMapping =
            arrus::getRange<ChannelIdx>(0, 192);

    [[nodiscard]] ProbeSettings toProbeSettings() const {
        return ProbeSettings{
                ProbeModel{modelId, numberOfElements, pitch, txFrequencyRange},
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
}

INSTANTIATE_TEST_CASE_P

(ValidProbeSettings, CorrectProbeSettingsTest,
 testing::Values(
         // 1-D, all channels
         TestProbeSettings{},
         // 1-D, subset of the underlying adapter is used
         TestProbeSettings{
                 .numberOfElements{96},
                 .channelMapping = arrus::concat(
                         getRange<ChannelIdx>(0, 48),
                         getRange<ChannelIdx>(144, 192)
                 )
         },
         // 2-D, all channels are used
         TestProbeSettings{
                 .numberOfElements{8, 8},
                 .pitch {.3e-3, .3e-3},
                 .channelMapping = getRange<ChannelIdx>(0, 64)
         },
         // 2-D, some of the channels are used
         TestProbeSettings{
                 .numberOfElements{16, 16},
                 .pitch {.3e-3, .3e-3},
                 .channelMapping = arrus::concat(
                         getRange<ChannelIdx>(0, 128),
                         getRange<ChannelIdx>(512, 640)
                 )
         }
 ));


class IncorrectProbeSettingsTest
        : public testing::TestWithParam<TestProbeSettings> {
};

TEST_P(IncorrectProbeSettingsTest, RejectsIncorrect) {
    ProbeSettingsValidator validator(0);
    TestProbeSettings val = GetParam();
    validator.validate(val.toProbeSettings());
    EXPECT_THROW(validator.throwOnErrors(), IllegalArgumentException);
}

INSTANTIATE_TEST_CASE_P

(InvalidProbeSettings, IncorrectProbeSettingsTest,
 testing::Values(
         // Testing probe model
         // - Negative pitch
         TestProbeSettings{
                 .pitch{-.3e-3}
         },
         // - Negative frequency
         TestProbeSettings{
                 .txFrequencyRange{-10e6, 10e6}
         },
         // - Invalid number of channels in mapping
         TestProbeSettings{
                 .numberOfElements{96},
                 .channelMapping = getRange<ChannelIdx>(0, 192)
         },
         // - Non-unique channels
         TestProbeSettings{
                 .numberOfElements{32},
                 .channelMapping = ::arrus::concat(
                         getRange<ChannelIdx>(0, 2),
                         getRange<ChannelIdx>(0, 30)
                 )
         }
 ));

}

