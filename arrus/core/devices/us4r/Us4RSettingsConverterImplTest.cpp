#include <gtest/gtest.h>

#include <type_traits>
#include <ostream>

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
using ChannelMapping = ProbeAdapterSettings::ChannelMapping;

// -------- Mappings

std::vector<ChannelIdx> generateReversed(ChannelIdx a, ChannelIdx b) {
    std::vector<ChannelIdx> result;
    for(int i = b-1; i >= a; --i) {
        result.push_back(i);
    }
    return result;
}

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
            mappings.adapterMapping);

    RxSettings rxSettings({}, 24, 24, {}, 10e6, {});

    auto[us4oemSettings, newAdapterSettings] = converter.convertToUs4OEMSettings(
        adapterSettings, rxSettings,
        Us4OEMSettings::ReprogrammingMode::SEQUENTIAL, std::nullopt, {}, 2);

    EXPECT_EQ(us4oemSettings.size(), mappings.expectedUs4OEMMappings.size());

    for(int i = 0; i < us4oemSettings.size(); ++i) {
        EXPECT_EQ(us4oemSettings[i].getChannelMapping(), mappings.expectedUs4OEMMappings[i]);
    }

    EXPECT_EQ(newAdapterSettings.getChannelMapping(), mappings.expectedAdapterMapping);
}

INSTANTIATE_TEST_CASE_P

(TestingMappings, MappingsTest,
 testing::Values(
         // NOTE: the assumption is here that unmapped us4oem channels
         // are mapped by identity function.
         // Two modules, 0: 0-32, 1: 0-32
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{Ordinal(i / 32), ChannelIdx(i % 32)};
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
                     return ChannelAddress{Ordinal(i / 32), ChannelIdx(i % 32)};
                 })
         )),
// Two modules, 0: 0-128, 1: 0-64
         ARRUS_STRUCT_INIT_LIST(Mappings, (
                 x.adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{Ordinal(i / 128), ChannelIdx(i % 128)};
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
                     return ChannelAddress{Ordinal(i / 128), ChannelIdx(i % 128)};
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
                                         generateReversed(32, 64),
                                         getRange<ChannelIdx>(64, 96),
                                         generateReversed(96, 128)
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
                                        generateReversed(0, 32),
                                        generateReversed(32, 64),
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
                                     generateReversed(0, 32),
                                     generateReversed(32, 64),
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
                     return ChannelAddress{Ordinal(i % 2), ChannelIdx(i / 2)};
                 }),
                 x.expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
                 x.expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{Ordinal(i % 2), ChannelIdx(i / 2)};
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
                     return ChannelAddress{Ordinal(i % 2), ChannelIdx(i / 2)};
                 })
         ))
 ));

std::vector<ChannelIdx> getSL1543ChannelMapping() {
    return ::arrus::getRange<ChannelIdx>(0, 192);
}

std::vector<ChannelIdx> getAtlLikeProbeChannelMapping() {
    return ::arrus::getRange<ChannelIdx>(0, 128);
}

std::vector<ChannelIdx> getEsaotePhaseArrayProbeMapping() {
    std::vector<ChannelIdx> result;
    for(int i = 0; i < 48; ++i) {
        result.push_back(i);
    }
    for(int i = 144; i < 192; ++i) {
        result.push_back(i);
    }
    return result;
}

std::vector<ChannelIdx> getOneByOneProbeMapping() {
    return ::arrus::getRange<ChannelIdx>(0, 128);
}

ProbeAdapterSettings::ChannelMapping getEsaote3ChannelMapping() {
    ProbeAdapterSettings::ChannelMapping mapping;
    for(int i = 0; i < 192; ++i) {
        auto group = i / 32;
        auto module = group % 2;
        auto channel = i % 32 + 32*(i/64);
        mapping.push_back({module, channel});
    }
    return mapping;
}

ProbeAdapterSettings::ChannelMapping getEsaote3Us4RChannelMapping() {
    // 6 modules used
    ProbeAdapterSettings::ChannelMapping mapping;
    for(int i = 0; i < 192; ++i) {
        auto group = i / 32;
        auto module = group;
        auto channel = i % 32;
        mapping.push_back({module, channel});
    }
    return mapping;
}

ProbeAdapterSettings::ChannelMapping getAtlUs4RLikeChannelMapping() {
    // 6 modules used
    ProbeAdapterSettings::ChannelMapping mapping;
    for(int i = 0; i < 128; ++i) {
        auto group = i / 32;
        auto module = group;
        auto channel = i % 32;
        mapping.push_back({module, channel});
    }
    return mapping;
}

ProbeAdapterSettings::ChannelMapping getOneByOneChannelMapping() {
    ProbeAdapterSettings::ChannelMapping mapping;
    for(int i = 0; i < 128; ++i) {
        mapping.emplace_back(i%2, i / 2);
    }
    return mapping;
}

TEST(Us4OEMRemappingTest, CorrectlyRemapsUs4OEMNumbersEsaote) {
    Us4RSettingsConverterImpl converter;

    auto probeMapping = getSL1543ChannelMapping();
    auto adapterMapping = getEsaote3Us4RChannelMapping();
    ChannelIdx nChannels = probeMapping.size();

    ProbeAdapterSettings adapterSettings(
        ProbeAdapterModelId("test", "test"),
        adapterMapping.size(),
        adapterMapping
    );
    RxSettings rxSettings({}, 24, 24, {}, 10e6, {});
    Ordinal nUs4OEMs = 8;
    std::vector<Ordinal> adapterToUs4R = {
        /* Adapter Us4OEM: 0 -> Us4R Us4OEM: */ 0,
        /* Adapter Us4OEM: 1 -> US4R Us4OEM: */ 1,
        /* Adapter Us4OEM: 2 -> Us4R Us4OEM: */ 2,
        /* Adapter Us4OEM: 3 -> Us4R Us4OEM: */ 4,
        /* Adapter Us4OEM: 4 -> Us4R Us4OEM: */ 5,
        /* Adapter Us4OEM: 5 -> Us4R Us4OEM: */ 6,
    };

    auto [us4oemCfg, adapterCfg] = converter.convertToUs4OEMSettings(
        adapterSettings, rxSettings,
        Us4OEMSettings::ReprogrammingMode::SEQUENTIAL, nUs4OEMs, adapterToUs4R, 2);
    // Adapter mapping
    auto &actualAdapterMapping = adapterCfg.getChannelMapping();
    ChannelMapping expectedAdapterMapping = concat<ChannelAddress>(
        {
            zip(getNTimes<Ordinal>(0, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(1, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(2, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(4, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(5, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(6, 32), getRange<ChannelIdx>(0, 32)),
        }
    );
    // Probe adapter mapping.
    ASSERT_EQ(actualAdapterMapping, expectedAdapterMapping);
    // Us4OEM mapping.
    for(auto &us4oem: us4oemCfg) {
        ASSERT_EQ(us4oem.getChannelMapping(), getRange<ChannelIdx>(0, 128));
    }
}

TEST(Us4OEMRemappingTest, CorrectlyRemapsUs4OEMNumbersAtl) {
    Us4RSettingsConverterImpl converter;

    auto probeMapping = getAtlLikeProbeChannelMapping();
    auto adapterMapping = getAtlUs4RLikeChannelMapping();
    ChannelIdx nChannels = probeMapping.size();

    ProbeAdapterSettings adapterSettings(
        ProbeAdapterModelId("test", "test"),
        adapterMapping.size(),
        adapterMapping
    );
    RxSettings rxSettings({}, 24, 24, {}, 10e6, {});
    Ordinal nUs4OEMs = 8;
    std::vector<Ordinal> adapterToUs4R = {
        /* Adapter Us4OEM: 0 -> Us4R Us4OEM: */ 0,
        /* Adapter Us4OEM: 1 -> US4R Us4OEM: */ 2,
        /* Adapter Us4OEM: 2 -> Us4R Us4OEM: */ 5,
        /* Adapter Us4OEM: 3 -> Us4R Us4OEM: */ 7,
    };

    auto [us4oemCfg, adapterCfg] = converter.convertToUs4OEMSettings(
        adapterSettings, rxSettings,
        Us4OEMSettings::ReprogrammingMode::SEQUENTIAL, nUs4OEMs, adapterToUs4R, 2);
    // Adapter mapping
    auto &actualAdapterMapping = adapterCfg.getChannelMapping();
    ChannelMapping expectedAdapterMapping = concat<ChannelAddress>(
        {
            zip(getNTimes<Ordinal>(0, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(2, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(5, 32), getRange<ChannelIdx>(0, 32)),
            zip(getNTimes<Ordinal>(7, 32), getRange<ChannelIdx>(0, 32)),
        }
    );
    // Probe adapter mapping.
    ASSERT_EQ(actualAdapterMapping, expectedAdapterMapping);
    // Us4OEM mapping.
    for(auto &us4oem: us4oemCfg) {
        ASSERT_EQ(us4oem.getChannelMapping(), getRange<ChannelIdx>(0, 128));
    }
}

}

