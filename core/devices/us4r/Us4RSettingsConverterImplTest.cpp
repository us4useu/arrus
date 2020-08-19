#include <gtest/gtest.h>

#include <type_traits>
#include <ostream>

#include "arrus/core/devices/us4r/Us4RSettingsConverterImpl.h"

#include "arrus/core/common/collections.h"
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"
#include "arrus/core/api/devices/probe/ProbeSettings.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"


namespace {

using namespace arrus;

using ChannelAddress = ProbeAdapterSettings::ChannelAddress;

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
         Mappings{
                 .adapterMapping = generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i / 32, i % 32};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
// The same as the input adapter mapping
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i / 32, i % 32};
                 })
         },
// Two modules, 0: 0-128, 1: 0-64
         Mappings{
                 .
                 adapterMapping = generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
// The same as the input adapter mapping
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(192, [](size_t i) {
                     return ChannelAddress{i / 128, i % 128};
                 })
         },
// Two modules, 0: 0-31, 1: 63-32, 0: 64-95, 1: 127-96
// Expected: us4oems: 0: 0-127, 1: 0-31, 63-32, 64-95, 127-96
// Expected: probe adapter: 0: 0-31, 1: 32-63, 0: 64-95, 1: 96-127
         Mappings{
                 .
                 adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 1) {
                         channel = (i / 32 + 1) * 32 - 1 - (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         getRange<ChannelIdx>(0, 32),
                                         ranges::views::iota(32, 64)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 96),
                                         ranges::views::iota(96, 128)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                 })
                 },
// The same as the input adapter mapping
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     return ChannelAddress{module, channel};
                 })
         },
// Two modules, 0: 32-0, 1: 0-32, 0: 64-32, 1: 32-64
         Mappings{
                 .
                 adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 0) {
                         channel = (i / 32 / 2 + 1) * 32 - 1 - (i % 32);
                     } else {
                         channel = (i / 32 / 2) * 32 + (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         ranges::views::iota(0, 32)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         ranges::views::iota(32, 64)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 128)
                                 }),
                         // 1:
                         getRange<ChannelIdx>(0, 128)

                 },
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = (i / 32 / 2) * 32 + (i % 32);
                     return ChannelAddress{module, channel};
                 })
         },
// two modules, groups are shuffled for us4oem: 0: 64-32, 32-0
         Mappings{
                 .
                 adapterMapping = generate<ChannelAddress>(128, [](size_t i) {
                     Ordinal module = (i / 32) % 2;
                     ChannelIdx channel = i;
                     if(module == 0) {
                         channel = (1- i / 32 / 2 + 1) * 32 - 1 - (i % 32);
                     } else {
                         channel = (i / 32 / 2) * 32 + (i % 32);
                     }
                     return ChannelAddress{module, channel};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         ::arrus::concat<ChannelIdx>(
                                 {
                                         ranges::views::iota(0, 32)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         ranges::views::iota(32, 64)
                                         | ranges::views::reverse
                                         | ranges::views::transform(
                                                 [](auto v) {
                                                     return ChannelIdx(v);
                                                 })
                                         | ranges::to_vector,
                                         getRange<ChannelIdx>(64, 128)
                                 }),
                         // 1:
                         getRange<ChannelIdx>(0, 128)

                 },
                 .
                 expectedAdapterMapping =
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
         },
// Two modules, 0: 0, 1: 0, 0: 1, 1: 1, ..., 0:31, 1:31
         Mappings{
                 .
                 adapterMapping = generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 }),
                 .
                 expectedUs4OEMMappings = {
                         // 0:
                         getRange<ChannelIdx>(0, 128),
                         // 1:
                         getRange<ChannelIdx>(0, 128)
                 },
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 })
         },
// Two modules, some randomly generated permutation
         Mappings{
                 .
                 adapterMapping =
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
                 .
                 expectedUs4OEMMappings = {
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
                 .
                 expectedAdapterMapping =
                 generate<ChannelAddress>(64, [](size_t i) {
                     return ChannelAddress{i % 2, i / 2};
                 })
         }
 ));


// Czy wlasciwe aktywne grupy kanalow sa ustawiane

}
// Czy prawidlowe wartosci Rx sa ustawiane

