#include <gtest/gtest.h>
#include <ostream>

#include "arrus/common/format.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/devices/us4r/validators/WatchdogSettingsValidator.h"

namespace {
using namespace arrus;
using namespace arrus::devices;

struct TestWatchdogSettings {
    WatchdogSettings settings;

    friend std::ostream &
    operator<<(std::ostream &os, const TestWatchdogSettings &settings) {
        os << "enabled: " << settings.settings.isEnabled()
           << " oemThreshold0: " << toString(settings.settings.getOEMThreshold0())
           << " oemThreshold1: " << toString(settings.settings.getOEMThreshold1())
           << " hostThreshold: " << toString(settings.settings.getHostThreshold());
        return os;
    }
};

class CorrectWatchdogSettingsTest
    : public testing::TestWithParam<TestWatchdogSettings> {
};

TEST_P(CorrectWatchdogSettingsTest, AcceptsCorrect) {
    WatchdogSettingsValidator validator;
    TestWatchdogSettings val = GetParam();
    validator.validate(val.settings);
    EXPECT_NO_THROW(validator.throwOnErrors());
    validator.throwOnErrors();
}

INSTANTIATE_TEST_CASE_P

    (ValidWatchdogSettings, CorrectWatchdogSettingsTest,
     testing::Values(
         // 1-D, all channels
         TestWatchdogSettings{WatchdogSettings::disabled()},
         // All thresholds equal.
         TestWatchdogSettings{WatchdogSettings{2.0f, 2.0f, 2.0f}},
         // Threshold OEM 0 < Threshold OEM 1
         TestWatchdogSettings{WatchdogSettings{1.9f, 2.0f, 2.0f}},
         // Max thresholds
         TestWatchdogSettings{WatchdogSettings{8.125f, 8.125f, 8.125f}}
    ));


class IncorrectProbeSettingsTest
    : public testing::TestWithParam<TestWatchdogSettings> {
};

TEST_P(IncorrectProbeSettingsTest, RejectsIncorrect) {
    WatchdogSettingsValidator validator;
    TestWatchdogSettings val = GetParam();
    validator.validate(val.settings);
    EXPECT_THROW(validator.throwOnErrors(), ::arrus::IllegalArgumentException);
}

INSTANTIATE_TEST_CASE_P

    (InvalidProbeSettings, IncorrectProbeSettingsTest,
     testing::Values(
         // OEM 0 too low value
         TestWatchdogSettings{WatchdogSettings{1e-4f, 1.0f, 2.0f}},
         // OEM 1 too high value
         TestWatchdogSettings{WatchdogSettings{1.0f, 8.193f, 2.0f}},
         // OEM 0 > OEM 1
         TestWatchdogSettings{WatchdogSettings{1.1f, 1.0f, 2.0f}},
         // host watchdog too low
         TestWatchdogSettings{WatchdogSettings{1.1f, 2.0f, 1e-4f}},
         // host watchdog too low
         TestWatchdogSettings{WatchdogSettings{1.1f, 2.0f, 8.193f}}
     ));

}


