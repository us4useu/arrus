#include <gtest/gtest.h>

#include <type_traits>
#include <ostream>
#include <range/v3/all.hpp>

#include "arrus/core/devices/us4r/Us4RSettingsConverterImpl.h"

#include "arrus/core/common/tests.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"


namespace {

using namespace arrus;
using namespace arrus::devices;

using ChannelAddress = ProbeAdapterSettings::ChannelAddress;

// -------- Mappings

struct Mappings {
    ProbeAdapterSettings::ChannelMapping adapterMapping;

    std::vector<std::vector<ChannelIdx>> expectedUs4OEMMappings;
    ProbeAdapterSettings::ChannelMapping expectedAdapterMapping;

    friend std::ostream &
    operator<<(std::ostream &os, const Mappings &mappings) {
        os << "adapterMapping: ";
        for(const auto &address : mappings.adapterMapping) {
            os << "(" << address.first << ", " << address.second << ") ";
        }

        os << "expectedOs4OEMMappings: ";

        int i = 0;
        for(const auto &mapping: mappings.expectedUs4OEMMappings) {
            os << "mapping for Us4OEM: " << i++ << " :";
            for(auto v : mapping) {
                os << v << " ";
            }
        }

        os << "expectedAdapterMapping: ";

        for(const auto &address : mappings.expectedAdapterMapping) {
            os << "(" << address.first << ", " << address.second << ") ";
        }
        return os;
    }
};

class MappingsTest
        : public testing::TestWithParam<Mappings> {
};

TEST_P(MappingsTest, CorrectlyConvertsMappingsToUs4OEMSettings) {
    Us4RSettingsConverterImpl converter;

    Mappings mappings = GetParam();

    ProbeAdapterSettings adapterSettings(
            ProbeAdapterModelId("test", "test"),
            mappings.adapterMapping.size(),
            mappings.adapterMapping
    );

    ProbeSettings probeSettings(
            ProbeModel(ProbeModelId("test", "test"),
                       {32},
                       {0.3e-3}, {1e6, 10e6}),
            getRange<ChannelIdx>(0, 32)
    );

    RxSettings rxSettings({}, 24, 24, {}, 10e6, {});

    auto[us4oemSettings, newAdapterSettings] =
    converter.convertToUs4OEMSettings(adapterSettings, probeSettings,
                                      rxSettings);

//    std::cerr << "output probe adapter settings: " << std::endl;
//    for(auto [m, ch] : newAdapterSettings.getChannelMapping()) {
//        std::cerr << "(" << m << "," << ch << ")";
//    }
//    std::cerr << std::endl;

    EXPECT_EQ(us4oemSettings.size(), mappings.expectedUs4OEMMappings.size());

    for(int i = 0; i < us4oemSettings.size(); ++i) {
        EXPECT_EQ(us4oemSettings[i].getChannelMapping(),
                  mappings.expectedUs4OEMMappings[i]);
    }

    EXPECT_EQ(newAdapterSettings.getChannelMapping(),
              mappings.expectedAdapterMapping);
}

INSTANTIATE_TEST_CASE_P

(TestingMappings, MappingsTest,
 testing::Values(
         // NOTE: the assumption is here that unmapped us4oem channels
         // are mapped by identity function.
         // Two modules, 0: 0-32, 1: 0-32
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i / 32, i % 32};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
// The same as the input adapter mapping
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i / 32, i % 32};
                 })
         )),
// Two modules, 0: 0-128, 1: 0-64
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
// The same as the input adapter mapping
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 })
         )),
// Two modules, 0: 0-31, 1: 63-32, 0: 64-95, 1: 127-96
// Expected: us4oems: 0: 0-127, 1: 0-31, 63-32, 64-95, 127-96
// Expected: probe adapter: 0: 0-31, 1: 32-63, 0: 64-95, 1: 96-127
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 1) {
                         channel = (i / 32 + 1) * 32 - 1 - (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         getRange<ChannelIdx>(0, 32),
                                         ranges::view::iota(32, 64)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 96),
                                         ranges::view::iota(96, 128)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                 })
                 },
// The same as the input adapter mapping
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     return ChannelAddress{module, channel};
                 })
         )),
// Two modules, 0: 32-0, 1: 0-32, 0: 64-32, 1: 32-64
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 0) {
                         channel = (i / 32 / 2 + 1) * 32 - 1 - (i % 32);
                     } else {
                         channel = (i / 32 / 2) * 32 + (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         ranges::view::iota(0, 32)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         ranges::view::iota(32, 64)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 128)
                                 }),
                         // 1:
                         getRange<ChannelIdx>(0, 128)

                 },
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = (i / 32 / 2) * 32 + (i % 32);
                     return ChannelAddress{module, channel};
                 })
         )),
// two modules, groups are shuffled for us4oem: 0: 64-32, 32-0
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 0) {
                         channel = (1- i / 32 / 2 + 1) * 32 - 1 - (i % 32);
                     } else {
                         channel = (i / 32 / 2) * 32 + (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         ranges::view::iota(0, 32)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         ranges::view::iota(32, 64)
                                         | ranges::view::reverse
                                         | ranges::view::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 128)
                                 }),
                         // 1:
                         getRange<ChannelIdx>(0, 128)

                 },
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(128, [](size_t i)
                 {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 0) {
                         channel = (1- i / 32 / 2) * 32 + (i % 32);
                     } else {
                         channel = (i / 32 / 2) * 32 + (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 })
         )),
// Two modules, 0: 0, 1: 0, 0: 1, 1: 1, ..., 0:31, 1:31
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 })
         )),
// Two modules, some randomly generated permutation
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping =
// Random mapping for modules 0, 1, 0, 1...
                         {{0, 17},
                          {1, 13},
                          {0, 11},
                          {1, 17},
                          {0, 21},
                          {1, 16},
                          {0, 20},
                          {1, 1},
                          {0, 9},
                          {1, 23},
                          {0, 26},
                          {1, 20},
                          {0, 5},
                          {1, 12},
                          {0, 16},
                          {1, 21},
                          {0, 10},
                          {1, 14},
                          {0, 28},
                          {1, 5},
                          {0, 15},
                          {1, 31},
                          {0, 2},
                          {1, 9},
                          {0, 12},
                          {1, 26},
                          {0, 8},
                          {1, 22},
                          {0, 18},
                          {1, 19},
                          {0, 23},
                          {1, 3},
                          {0, 29},
                          {1, 7},
                          {0, 13},
                          {1, 4},
                          {0, 0},
                          {1, 18},
                          {0, 19},
                          {1, 15},
                          {0, 6},
                          {1, 27},
                          {0, 24},
                          {1, 28},
                          {0, 4},
                          {1, 11},
                          {0, 27},
                          {1, 0},
                          {0, 30},
                          {1, 2},
                          {0, 31},
                          {1, 10},
                          {0, 25},
                          {1, 29},
                          {0, 14},
                          {1, 25},
                          {0, 1},
                          {1, 24},
                          {0, 3},
                          {1, 6},
                          {0, 22},
                          {1, 30},
                          {0, 7},
                          {1, 8}
                         },
                 x.expectedUs4OEMMappings = {
                         // 0:
                         ::arrus::concat<ChannelIdx>(
                                {
                                      std::vector<ChannelIdx>({
                                         17, 11, 21, 20,
                                         9, 26, 5, 16,
                                         10, 28, 15, 2,
                                         12, 8, 18, 23,
                                         29, 13, 0, 19,
                                         6, 24, 4, 27,
                                         30, 31, 25, 14,
                                         1, 3, 22, 7
                                      }),
                                      ::arrus::getRange<ChannelIdx>(32, 128)
                         }),
                         // 1:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                     std::vector<ChannelIdx> {
                                         13, 17, 16,  1,
                                         23, 20, 12, 21,
                                         14,  5, 31,  9,
                                         26, 22, 19,  3,
                                         7, 4, 18, 15,
                                         27, 28, 11,  0,
                                         2, 10, 29, 25,
                                         24,  6, 30,  8
                                     },
                                     ::arrus::getRange<ChannelIdx>(32, 128)
                         })
                 },
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 })
         ))
 ));

// -------- Groups of active channels

struct ActiveChannels {
    ProbeAdapterSettings::ChannelMapping adapterMapping;
    std::vector<ChannelIdx> probeMapping;

    std::vector<BitMask> expectedUs4OEMMasks;

    friend std::ostream &
    operator<<(std::ostream &os, const ActiveChannels &mappings) {
        os << "adapterMapping: ";
        for(const auto &address : mappings.adapterMapping) {
            os << "(" << address.first << ", " << address.second << ") ";
        }

        os << "probeMapping: ";

        for(auto value : mappings.probeMapping) {
            os << value << " ";
        }

        os << "expected groups masks: ";

        int i = 0;
        for(const auto & mask: mappings.expectedUs4OEMMasks) {
            os << "Us4OEM:" << i << " :";
            for(auto value : mask) {
                os << (int) value << " ";
            }
        }
        return os;
    }
};

class ActiveChannelsTest
        : public testing::TestWithParam<ActiveChannels> {
};

TEST_P(ActiveChannelsTest, CorrectlyGeneratesActiveChannelGroups) {
    Us4RSettingsConverterImpl converter;

    ActiveChannels testCase = GetParam();

    ProbeAdapterSettings adapterSettings(
            ProbeAdapterModelId("test", "test"),
            testCase.adapterMapping.size(),
            testCase.adapterMapping
    );

    ProbeSettings probeSettings(
            ProbeModel(ProbeModelId("test", "test"),
                       {32},
                       {0.3e-3}, {1e6, 10e6}),
            testCase.probeMapping
    );

    RxSettings rxSettings({}, 24, 24, {}, 10e6, {});

    auto[us4oemSettings, newAdapterSettings] =
    converter.convertToUs4OEMSettings(adapterSettings, probeSettings,
                                      rxSettings);

    EXPECT_EQ(us4oemSettings.size(), testCase.expectedUs4OEMMasks.size());

    for(int i = 0; i < us4oemSettings.size(); ++i) {
        EXPECT_EQ(us4oemSettings[i].getActiveChannelGroups(),
                  testCase.expectedUs4OEMMasks[i]);
    }
}

INSTANTIATE_TEST_CASE_P

(TestingActiveChannelGroups, ActiveChannelsTest,
 testing::Values(
// Esaote 1 like case, full adapter to probe mapping
// us4oem:0 :0-128, us4oem:1 : 0-64
         ARRUS_STRUCT_INIT_LIST(ActiveChannels, (
                 x.adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 }),
                 x.probeMapping = getRange<ChannelIdx>(0, 192),
                 x.expectedUs4OEMMasks = {
                         // Us4OEM: 0
                         getNTimes<bool>(true, 16),
                         // Us4OEM: 1
                         ::arrus::concat<bool>({
                            getNTimes<bool>(true, 8),
                            getNTimes<bool>(false, 8)
                         })
                 }
         )),
// Esaote 1 case, partial adapter to probe mapping
         ARRUS_STRUCT_INIT_LIST(ActiveChannels, (
                 x.adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 }),
                 x.probeMapping = ::arrus::concat<ChannelIdx>({
                     getRange<ChannelIdx>(0, 48),
                     getRange<ChannelIdx>(144, 192),
                 }),
                 x.expectedUs4OEMMasks = {
                         // Us4OEM: 0
                         ::arrus::concat<bool>({
                            getNTimes<bool>(true, 6),
                            getNTimes<bool>(false, 10)
                         }),
                         // Us4OEM: 1
                         ::arrus::concat<bool>({
                             getNTimes<bool>(false, 2),
                             getNTimes<bool>(true, 6),
                             getNTimes<bool>(false, 8)
                         })
                 }
         )),
// esaote 1, but reverse the channels: 32-0, 64-32, .. for module 0;
// for module 1 keep the order as is
// partial adapter to probe mapping
    ARRUS_STRUCT_INIT_LIST(ActiveChannels, (
        x.adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
            Ordinal module;
            ChannelIdx channel;
            if(i < 128) {
                ChannelIdx group = i / 32;
                module = 0;
                channel = (group+1) * 32 - (i % 32 + 1);
            } else {
                module = 1;
                channel = i % 128;
            }
            return ChannelAddress{module, channel};
        }),
        x.probeMapping = ::arrus::concat<ChannelIdx>({
                getRange<ChannelIdx>(0, 48),
                getRange<ChannelIdx>(144, 192)
        }),
        x.expectedUs4OEMMasks = {
            // Us4OEM: 0
            ::arrus::concat<bool>({
                getNTimes<bool>(true, 4),
                getNTimes<bool>(false, 2),
                getNTimes<bool>(true, 2),
                getNTimes<bool>(false, 8)
            }),
            // Us4OEM: 1
            ::arrus::concat<bool>({
                getNTimes<bool>(false, 2),
                getNTimes<bool>(true, 6),
                getNTimes<bool>(false, 8)
            })
        }
    ))
));
}

