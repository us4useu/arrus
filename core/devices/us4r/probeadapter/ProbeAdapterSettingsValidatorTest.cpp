#include <gtest/gtest.h>
#include <iostream>
#include <utility>
#include <range/v3/view/zip.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/all.hpp>

#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"

namespace {

using namespace arrus;
using ChannelAddress = ::arrus::ProbeAdapterSettings::ChannelAddress;

struct TestAdapterSettings {
    arrus::ProbeAdapterModelId modelId{"test", "test"};
    ChannelIdx nChannels = 64;
    ProbeAdapterSettings::ChannelMapping channelMapping;

    [[nodiscard]] ProbeAdapterSettings toProbeAdapterSettings() const {
        return ProbeAdapterSettings(modelId, nChannels, channelMapping);
    }

    friend std::ostream &
    operator<<(std::ostream &os, const TestAdapterSettings &settings) {

        std::function<std::string(ChannelAddress)> func = [](ChannelAddress v) {
            return "(" +
                   std::to_string(v.first) +
                   ", " +
                   std::to_string(v.second) + ")";
        };
        os << " nChannels: " << settings.nChannels
           << " channelMapping: " <<
           (arrus::toStringTransform(settings.channelMapping, func));
        return os;
    }

};

class CorrectAdapterSettingsTest
        : public testing::TestWithParam<TestAdapterSettings> {
};

TEST_P(CorrectAdapterSettingsTest, AcceptsCorrect) {
    ProbeAdapterSettingsValidator validator(0);
    TestAdapterSettings val = GetParam();
    validator.validate(val.toProbeAdapterSettings());
    EXPECT_NO_THROW(validator.throwOnErrors());
}

INSTANTIATE_TEST_CASE_P

(ValidProbeAdapterSettings, CorrectAdapterSettingsTest,
 testing::Values(
         // at given position i of the probe adapter:
         // (us4oem ordinal, us4oem channel)
         // (0, 0), (1, 0), (0, 1), (1, 1),... (0, 63), (1, 63)
         TestAdapterSettings{
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             std::cout << i % 2 << ", " << i / 2 << std::endl;
                             return std::make_pair(i % 2, i / 2);
                         })
         },
         // (0, 0), (1, 0), (0, 1), (1, 1),... (0, 127), (1, 127)
         TestAdapterSettings{
                 .nChannels = 128,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         128, [](size_t i) {
                             std::cout << i % 2 << ", " << i / 2 << std::endl;
                             return std::make_pair(i % 2, i / 2);
                         })
         },
         // reverse channels
         // (0, 63), (1, 63), (0, 62), (1, 62),..., (0, 0), (1, 0)
         TestAdapterSettings{
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             auto res = std::make_pair(i % 2, 63 - i / 2);
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         },
         // reverse modules
         // (1, 0), (0, 0), (1, 1), (0, 1),... (1, 63), (0, 63)
         TestAdapterSettings{
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             auto res = std::make_pair(1 - i % 2, i / 2);
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         },
         // some non-trivial groups
         // us4oem:0: [0-32), [64-96)
         // us4oem:1: [32-64), [96-128)
         TestAdapterSettings{
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             Ordinal module;
                             if((i / 32) % 2 == 0) {
                                 module = 0;
                             } else {
                                 module = 1;
                             }
                             auto res = std::make_pair(module, i);
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         },
         // single module
         TestAdapterSettings{
                 .nChannels = 128,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         128, [](size_t i) {
                             return std::make_pair(0, i);
                         })
         },
         // 8 modules
         TestAdapterSettings{
                 .nChannels = 256,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         256, [](size_t i) {
                             return std::make_pair(i % 8, i / 8);
                         })
         }
 ));


class IncorrectAdapterSettingsTest
        : public testing::TestWithParam<TestAdapterSettings> {
};

TEST_P(IncorrectAdapterSettingsTest, RejectIncorrect) {
    ProbeAdapterSettingsValidator validator(0);
    TestAdapterSettings val = GetParam();
    validator.validate(val.toProbeAdapterSettings());
    EXPECT_THROW(validator.throwOnErrors(), IllegalArgumentException);
    try {
        validator.throwOnErrors();
    } catch(const IllegalArgumentException &e) {
        std::cerr << "The exception message: " << e.what() << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P

(InvalidProbeAdapterSettings, IncorrectAdapterSettingsTest,
 testing::Values(
         // Invalid size of the mapping - should be the same as nChannels
         TestAdapterSettings{
                 .nChannels=128,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             return std::make_pair(i % 2, i / 2);
                         })
         },
         // Invalid us4oem ordinal numbers: should be consecutive
         // Configuring us4oems 0 and 2
         TestAdapterSettings{
                 .nChannels=64,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             return std::make_pair(2 * (i % 2), i / 2);
                         })
         },
         // One of the us4oems has incomplete (not divisible by number of Rx
         // channels) mapping
         TestAdapterSettings{
                 .nChannels=48,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         48, [](size_t i) {
                             Ordinal module = i < 32 ? 0 : 1;
                             ChannelIdx channel = i % 32;
                             return std::make_pair(module, channel);
                         })
         },
         // One of the us4oems has non-unique set of group channels.
         TestAdapterSettings{
                 .nChannels=64,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             Ordinal module = i < 32 ? 0 : 1;
                             ChannelIdx channel;
                             if(i < 32) {
                                 // module 0: ok
                                 channel = i % 32;
                             } else {
                                 // module 1: channels 0, 1, 2, ..., 30, 0
                                 channel = i % 31;
                             }
                             return std::make_pair(module, channel);
                         })
         },
         // One of us4oems has mixed group of channels
         TestAdapterSettings{
                 .nChannels=64,
                 .channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             Ordinal module = i < 32 ? 0 : 1;
                             ChannelIdx channel = i % 32;
                             // module 0: channels are ok
                             if(i >= 32) {
                                 // Module 1: channels 1, 2, ..., 33
                                 channel = (channel + 1) % 33;
                             }
                             return std::make_pair(module, channel);
                         })
         }
 ));


}


