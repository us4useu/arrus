#include <gtest/gtest.h>
#include <iostream>
#include <utility>

#include "arrus/core/common/tests.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"

namespace xyz {

using namespace ::arrus;
using namespace ::arrus::devices;
using ChannelAddress = ::arrus::devices::ProbeAdapterSettings::ChannelAddress;

struct TestAdapterSettings {
    ::arrus::devices::ProbeAdapterModelId modelId{"test", "test"};
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
           (arrus::toStringTransform<ProbeAdapterSettings::ChannelAddress>(
               settings.channelMapping, func));
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
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             std::cout << i % 2 << ", " << i / 2 << std::endl;
                             return std::make_pair(Ordinal(i % 2), ChannelIdx(i / 2));
                         })
        )),
         // (0, 0), (1, 0), (0, 1), (1, 1),... (0, 127), (1, 127)
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels = 128,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         128, [](size_t i) {
                             std::cout << i % 2 << ", " << i / 2 << std::endl;
                             return std::make_pair(Ordinal(i % 2), ChannelIdx(i / 2));
                         })
         )),
         // reverse channels
         // (0, 63), (1, 63), (0, 62), (1, 62),..., (0, 0), (1, 0)
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             auto res = std::make_pair(Ordinal(i % 2), ChannelIdx(63 - i / 2));
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         )),
         // reverse modules
         // (1, 0), (0, 0), (1, 1), (0, 1),... (1, 63), (0, 63)
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             auto res = std::make_pair(Ordinal(1 - i % 2), ChannelIdx(i / 2));
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         )),
         // some non-trivial groups
         // us4oem:0: [0-32), [64-96)
         // us4oem:1: [32-64), [96-128)
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             Ordinal module;
                             if((i / 32) % 2 == 0) {
                                 module = 0;
                             } else {
                                 module = 1;
                             }
                             auto res = std::make_pair(Ordinal(module), ChannelIdx(i));
                             std::cout << res.first << ", " << res.second
                                       << std::endl;
                             return res;
                         })
         )),
         // single module
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels = 128,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         128, [](size_t i) {
                             return std::make_pair(Ordinal(0), ChannelIdx(i));
                         })
         )),
         // 8 modules
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels = 256,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         256, [](size_t i) {
                             return std::make_pair(Ordinal(i % 8), ChannelIdx(i / 8));
                         })
         ))
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
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels=128,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             return std::make_pair(Ordinal(i % 2), ChannelIdx(i / 2));
                         })
         )),
         // One of the us4oems has incomplete (not divisible by number of Rx
         // channels) mapping
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels=48,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         48, [](size_t i) {
                             Ordinal module = i < 32 ? 0 : 1;
                             ChannelIdx channel = i % 32;
                             return std::make_pair(Ordinal(module), ChannelIdx(channel));
                         })
         )),
         // One of the us4oems has non-unique set of group channels.
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels=64,
                 x.channelMapping = arrus::generate<ChannelAddress>(
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
                             return std::make_pair(Ordinal(module), ChannelIdx(channel));
                         })
         )),
         // One of us4oems has mixed group of channels
         ARRUS_STRUCT_INIT_LIST(TestAdapterSettings, (
                 x.nChannels=64,
                 x.channelMapping = arrus::generate<ChannelAddress>(
                         64, [](size_t i) {
                             Ordinal module = i < 32 ? 0 : 1;
                             ChannelIdx channel = i % 32;
                             // module 0: channels are ok
                             if(i >= 32) {
                                 // Module 1: channels 1, 2, ..., 33
                                 channel = (channel + 1) % 33;
                             }
                             return std::make_pair(Ordinal(module), ChannelIdx(channel));
                         })
         ))
 ));


}


