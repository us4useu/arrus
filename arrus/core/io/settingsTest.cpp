#include <gtest/gtest.h>
#include <csignal>
#include <boost/filesystem.hpp>

#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/io/settings.h"
#include "arrus/core/common/collections.h"

using namespace ::arrus;
using namespace ::arrus::session;
using namespace ::arrus::devices;
using namespace arrus::ops::us4r;

// ARRUS_TEST_DATA_PATH is defined in cmake.

std::vector<ChannelIdx> generateReversed(ChannelIdx a, ChannelIdx b) {
    std::vector<ChannelIdx> result;
    for(int i = b - 1; i >= a; --i) {
        result.push_back(i);
    }
    return result;
}

TEST(ReadingProtoTxtFile, readsUs4RPrototxtSettingsCorrectly) {
    auto filepath = boost::filesystem::path(ARRUS_TEST_DATA_PATH) /
                    boost::filesystem::path("us4r.prototxt");
    ::arrus::session::SessionSettings settings = arrus::io::readSessionSettings(
        filepath.string());
    auto const &us4rSettings = settings.getUs4RSettings();
    EXPECT_TRUE(us4rSettings.getUs4OEMSettings().empty());

    EXPECT_EQ(us4rSettings.getNumberOfUs4oems(), 2);
    std::vector<Ordinal> expectedAdapterToUs4RMapping = {1, 0};
    EXPECT_EQ(us4rSettings.getAdapterToUs4RModuleNumber(), expectedAdapterToUs4RMapping);

    EXPECT_EQ(us4rSettings.getChannelsMask(), std::vector<ChannelIdx>({}));
    EXPECT_EQ(us4rSettings.getUs4OEMChannelsMask(), std::vector<std::vector<uint8>>({{}, {}}));

    // Probe settings
    // Probe model
    auto const &probeSettings = us4rSettings.getProbeSettings();
    auto const &probeModel = probeSettings->getModel();
    EXPECT_EQ(probeModel.getModelId().getManufacturer(), "esaote");
    EXPECT_EQ(probeModel.getModelId().getName(), "sl1543");
    // 1-d, linear array.
    EXPECT_EQ(probeModel.getNumberOfElements().size(), 1);
    EXPECT_EQ(probeModel.getNumberOfElements()[0], 192);
    EXPECT_EQ(probeModel.getPitch().size(), 1);
    EXPECT_DOUBLE_EQ(probeModel.getPitch()[0], 0.245e-3);
    EXPECT_DOUBLE_EQ(probeModel.getTxFrequencyRange().start(), 1e6);
    EXPECT_DOUBLE_EQ(probeModel.getTxFrequencyRange().end(), 10e6);

    // Probe channel mapping
    EXPECT_EQ(probeSettings->getChannelMapping(), getRange<ChannelIdx>(0, 192));

    // Probe adapter settigns
    auto const &adapterSettings = us4rSettings.getProbeAdapterSettings();
    EXPECT_EQ(adapterSettings->getModelId().getManufacturer(), "us4us");
    EXPECT_EQ(adapterSettings->getModelId().getName(), "esaote2");
    EXPECT_EQ(adapterSettings->getNumberOfChannels(), 192);
    EXPECT_EQ(adapterSettings->getChannelMapping(),
              std::vector<ProbeAdapterSettings::ChannelAddress>(
                  {
                      { 0, 26 }, { 0, 27 }, { 0, 25 }, { 0, 23 }, { 0, 28 }, { 0, 22 }, { 0, 20 }, { 0, 21 },
                      { 0, 24 }, { 0, 18 }, { 0, 19 }, { 0, 15 }, { 0, 17 }, { 0, 16 }, { 0, 29 }, { 0, 13 },
                      { 0, 11 }, { 0, 14 }, { 0, 30 }, { 0, 8 }, { 0, 12 }, { 0, 5 }, { 0, 10 }, { 0, 9 },
                      { 0, 31 }, { 0, 7 }, { 0, 3 }, { 0, 6 }, { 0, 0 }, { 0, 2 }, { 0, 4 }, { 0, 1 },
                      { 1, 4 }, { 1, 3 }, { 1, 7 }, { 1, 5 }, { 1, 6 }, { 1, 2 }, { 1, 8 }, { 1, 9 },
                      { 1, 1 }, { 1, 11 }, { 1, 0 }, { 1, 10 }, { 1, 13 }, { 1, 12 }, { 1, 15 }, { 1, 14 },
                      { 1, 16 }, { 1, 17 }, { 1, 19 }, { 1, 18 }, { 1, 20 }, { 1, 25 }, { 1, 21 }, { 1, 22 },
                      { 1, 23 }, { 1, 31 }, { 1, 24 }, { 1, 27 }, { 1, 30 }, { 1, 26 }, { 1, 28 }, { 1, 29 },
                      { 0, 56 }, { 0, 55 }, { 0, 54 }, { 0, 53 }, { 0, 57 }, { 0, 52 }, { 0, 51 }, { 0, 49 },
                      { 0, 50 }, { 0, 48 }, { 0, 47 }, { 0, 46 }, { 0, 44 }, { 0, 45 }, { 0, 58 }, { 0, 42 },
                      { 0, 43 }, { 0, 59 }, { 0, 40 }, { 0, 41 }, { 0, 60 }, { 0, 38 }, { 0, 61 }, { 0, 39 },
                      { 0, 62 }, { 0, 34 }, { 0, 37 }, { 0, 63 }, { 0, 36 }, { 0, 35 }, { 0, 32 }, { 0, 33 },
                      { 1, 35 }, { 1, 34 }, { 1, 36 }, { 1, 38 }, { 1, 33 }, { 1, 37 }, { 1, 39 }, { 1, 40 },
                      { 1, 32 }, { 1, 41 }, { 1, 42 }, { 1, 43 }, { 1, 44 }, { 1, 45 }, { 1, 46 }, { 1, 47 },
                      { 1, 49 }, { 1, 48 }, { 1, 50 }, { 1, 52 }, { 1, 51 }, { 1, 55 }, { 1, 53 }, { 1, 54 },
                      { 1, 58 }, { 1, 56 }, { 1, 59 }, { 1, 57 }, { 1, 62 }, { 1, 61 }, { 1, 60 }, { 1, 63 },
                      { 0, 92 }, { 0, 93 }, { 0, 89 }, { 0, 91 }, { 0, 88 }, { 0, 90 }, { 0, 87 }, { 0, 85 },
                      { 0, 86 }, { 0, 84 }, { 0, 83 }, { 0, 82 }, { 0, 81 }, { 0, 80 }, { 0, 79 }, { 0, 77 },
                      { 0, 78 }, { 0, 76 }, { 0, 95 }, { 0, 75 }, { 0, 74 }, { 0, 94 }, { 0, 73 }, { 0, 72 },
                      { 0, 70 }, { 0, 64 }, { 0, 71 }, { 0, 68 }, { 0, 65 }, { 0, 69 }, { 0, 67 }, { 0, 66 },
                      { 1, 65 }, { 1, 67 }, { 1, 66 }, { 1, 69 }, { 1, 64 }, { 1, 68 }, { 1, 71 }, { 1, 70 },
                      { 1, 72 }, { 1, 74 }, { 1, 73 }, { 1, 75 }, { 1, 76 }, { 1, 77 }, { 1, 78 }, { 1, 79 },
                      { 1, 80 }, { 1, 82 }, { 1, 81 }, { 1, 83 }, { 1, 85 }, { 1, 84 }, { 1, 87 }, { 1, 86 },
                      { 1, 88 }, { 1, 92 }, { 1, 89 }, { 1, 94 }, { 1, 90 }, { 1, 91 }, { 1, 95 }, { 1, 93 }
                  }
              ));
    // Rx settings
    auto const &rxSettings = us4rSettings.getRxSettings();
    EXPECT_FALSE(rxSettings->getDtgcAttenuation().has_value());
    EXPECT_EQ(rxSettings->getPgaGain(), 30);
    EXPECT_EQ(rxSettings->getLnaGain(), 24);
    EXPECT_EQ(rxSettings->getTgcSamples(),
              std::vector<TGCSampleValue>({}));
    EXPECT_EQ(rxSettings->getLpfCutoff(), 15000000);
    EXPECT_EQ(rxSettings->getActiveTermination(), 200);
}

TEST(ReadingProtoTxtFile, readsCustomUs4RPrototxtSettingsCorrectly) {
    auto filepath = boost::filesystem::path(ARRUS_TEST_DATA_PATH) /
                    boost::filesystem::path("custom_us4r.prototxt");
    SessionSettings settings = arrus::io::readSessionSettings(
        filepath.string());
    auto const &us4rSettings = settings.getUs4RSettings();
    EXPECT_TRUE(us4rSettings.getUs4OEMSettings().empty());

    EXPECT_EQ(us4rSettings.getChannelsMask(), std::vector<ChannelIdx>({0, 15, 30}));
    EXPECT_EQ(us4rSettings.getUs4OEMChannelsMask(), std::vector<std::vector<uint8>>({{0, 15, 30},{}}));

    // Probe settings
    // Probe model
    auto const &probeSettings = us4rSettings.getProbeSettings();
    auto const &probeModel = probeSettings->getModel();
    EXPECT_EQ(probeModel.getModelId().getManufacturer(), "acme");
    EXPECT_EQ(probeModel.getModelId().getName(), "my_custom_probe");
    // 1-d, linear array.
    EXPECT_EQ(probeModel.getNumberOfElements().size(), 1);
    EXPECT_EQ(probeModel.getNumberOfElements()[0], 32);
    EXPECT_EQ(probeModel.getPitch().size(), 1);
    EXPECT_DOUBLE_EQ(probeModel.getPitch()[0], 0.21e-3);
    EXPECT_DOUBLE_EQ(probeModel.getTxFrequencyRange().start(), 1e6);
    EXPECT_DOUBLE_EQ(probeModel.getTxFrequencyRange().end(), 40e6);

    // Probe channel mapping
    EXPECT_EQ(probeSettings->getChannelMapping(),
              concat(getRange<ChannelIdx>(0, 16),
                     getRange<ChannelIdx>(48, 64)));

    // Probe adapter settigns
    auto const &adapterSettings = us4rSettings.getProbeAdapterSettings();
    EXPECT_EQ(adapterSettings->getModelId().getManufacturer(), "acme");
    EXPECT_EQ(adapterSettings->getModelId().getName(), "my_custom_adapter");
    EXPECT_EQ(adapterSettings->getNumberOfChannels(), 64);
    EXPECT_EQ(adapterSettings->getChannelMapping(),
              std::vector<ProbeAdapterSettings::ChannelAddress>(
                  {
                      { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 }, { 0, 2 }, { 1, 2 }, { 0, 3 }, { 1, 3 },
                      { 0, 4 }, { 1, 4 }, { 0, 5 }, { 1, 5 }, { 0, 6 }, { 1, 6 }, { 0, 7 }, { 1, 7 },
                      { 0, 8 }, { 1, 8 }, { 0, 9 }, { 1, 9 }, { 0, 10 }, { 1, 10 }, { 0, 11 }, { 1, 11 },
                      { 0, 12 }, { 1, 12 }, { 0, 13 }, { 1, 13 }, { 0, 14 }, { 1, 14 }, { 0, 15 }, { 1, 15 },
                      { 0, 16 }, { 1, 16 }, { 0, 17 }, { 1, 17 }, { 0, 18 }, { 1, 18 }, { 0, 19 }, { 1, 19 },
                      { 0, 20 }, { 1, 20 }, { 0, 21 }, { 1, 21 }, { 0, 22 }, { 1, 22 }, { 0, 23 }, { 1, 23 },
                      { 0, 24 }, { 1, 24 }, { 0, 25 }, { 1, 25 }, { 0, 26 }, { 1, 26 }, { 0, 27 }, { 1, 27 },
                      { 0, 28 }, { 1, 28 }, { 0, 29 }, { 1, 29 }, { 0, 30 }, { 1, 30 }, { 0, 31 }, { 1, 31 }
                  }
              ));
    // Rx settings
    auto const &rxSettings = us4rSettings.getRxSettings();
    EXPECT_EQ(rxSettings->getDtgcAttenuation(), 0);
    EXPECT_EQ(rxSettings->getPgaGain(), 24);
    EXPECT_EQ(rxSettings->getLnaGain(), 12);
    EXPECT_EQ(rxSettings->getTgcSamples(),
              std::vector<TGCSampleValue>({20, 21, 22}));
    EXPECT_EQ(rxSettings->getLpfCutoff(), 1000000);
    EXPECT_FALSE(rxSettings->getActiveTermination().has_value());
}

TEST(ReadingProtoTxtFile, readUs4OEMsPrototxtSettingsCorrectly) {
    auto filepath = boost::filesystem::path(ARRUS_TEST_DATA_PATH) /
                    boost::filesystem::path("us4oems.prototxt");
    SessionSettings settings = arrus::io::readSessionSettings(
        filepath.string());
    auto const &us4rSettings = settings.getUs4RSettings();
    EXPECT_FALSE(us4rSettings.getProbeAdapterSettings().has_value());
    EXPECT_FALSE(us4rSettings.getProbeSettings().has_value());
    EXPECT_FALSE(us4rSettings.getRxSettings().has_value());

    // us4oem:0
    auto const &us4oem0 = us4rSettings.getUs4OEMSettings()[0];
    EXPECT_EQ(us4oem0.getChannelMapping(),
              ::arrus::getRange<ChannelIdx>(0, 128));
    EXPECT_EQ(us4oem0.getActiveChannelGroups(), std::vector<bool>(
        {
            true, true, true, true,
                true, true, true, true,
                true, true, true, true,
                false, false, false, false
        }));
    // Rx settings
    auto const &rxSettings0 = us4oem0.getRxSettings();
    EXPECT_EQ(rxSettings0.getDtgcAttenuation(), 24);
    EXPECT_EQ(rxSettings0.getLnaGain(), 12);
    EXPECT_EQ(rxSettings0.getPgaGain(), 24);
    EXPECT_TRUE(rxSettings0.getTgcSamples().empty());
    EXPECT_EQ(rxSettings0.getLpfCutoff(), 10000000);
    EXPECT_FALSE(rxSettings0.getActiveTermination().has_value());

    // us4oem:1
    auto const &us4oem1 = us4rSettings.getUs4OEMSettings()[1];
    EXPECT_EQ(us4oem1.getChannelMapping(),
              generateReversed(0, 128));
    EXPECT_EQ(us4oem1.getActiveChannelGroups(), std::vector<bool>(
        {
            false, false, false, false,
                true, true, true, true,
                true, true, true, true,
                true, true, true, true
        }));
    // Rx settings
    auto const &rxSettings1 = us4oem1.getRxSettings();
    EXPECT_FALSE(rxSettings1.getDtgcAttenuation().has_value());
    EXPECT_EQ(rxSettings1.getLnaGain(), 30);
    EXPECT_EQ(rxSettings1.getPgaGain(), 24);
    EXPECT_EQ(rxSettings1.getTgcSamples(),
              std::vector<TGCSampleValue>({14.5, 15.5, 16.5, 17.5}));
    EXPECT_EQ(rxSettings1.getLpfCutoff(), 1000000);
    EXPECT_EQ(rxSettings1.getActiveTermination(), 500);
}

TEST(ReadingProtoTxtFile, throwsExceptionOnNoChannelsMask) {
    auto filepath = boost::filesystem::path(ARRUS_TEST_DATA_PATH) /
                    boost::filesystem::path("custom_us4r_no_channels_mask.prototxt");
    EXPECT_THROW(arrus::io::readSessionSettings(filepath.string()), IllegalArgumentException);
}

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

