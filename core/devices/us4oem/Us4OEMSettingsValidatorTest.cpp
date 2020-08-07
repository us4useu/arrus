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
    std::optional<uint8> dtgcAttenuation{100};
    uint8 pgaGain{30};
    uint8 lnaGain{24};
    uint32 lpfCutoff{(int) 10e6};
    std::optional<uint16> activeTermination{50};
    std::optional<TGCCurve> tgcSamples{getRange<float>(14, 54, 0.5)};

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
           << " pgaGain: " << (int)settings.pgaGain
           << " lnaGain: " << (int)settings.lnaGain
           << " lpfCutoff: " << settings.lpfCutoff
           << " activeTermination: " << toString(settings.activeTermination)
           << " tgcSamples: "
           << (settings.tgcSamples.has_value() ?
            toString(settings.tgcSamples.value()) :
               "(novalue)");
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
    validator.throwOnErrors();
//    EXPECT_NO_THROW(validator.throwOnErrors());
}

INSTANTIATE_TEST_CASE_P

(ValidUs4OEMSettings, CorrectUs4OEMSettingsTest,
 testing::Values(
         TestUs4OEMSettings{}
 ));

int main(int argc, char **argv) {
    INIT_ARRUS_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
}





