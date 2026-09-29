#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "Us4OEMDataTransferRegistrar.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace ::arrus::devices;
using namespace ::arrus::framework;

class Us4OEMDataTransferRegistrarTest : public ::testing::Test {
protected:
    void SetUp() override { Test::SetUp(); }

    Us4ROutputBuffer::SharedHandle createDstBuffer(const Us4OEMBuffer &buffer) {
        std::vector<Us4OEMBuffer> oemBuffers = {buffer};
        Us4ROutputBufferBuilder builder;
        return builder.setStopOnOverflow(false)
               .setNumberOfElements(1)
               .setLayoutTo(oemBuffers)
               .setUseP2pDma(false)
               .build();
    }

    Us4OEMBuffer createUs4OEMBuffer(const Us4OEMBufferArrayParts &parts) {
        Us4OEMBufferBuilder builder;
        size_t totalSize = std::accumulate(
            std::begin(parts), std::end(parts), size_t(0),
            [](const auto &a, const auto &b){
                return a + b.getSize();
            });
        const auto DATA_TYPE = Us4ROutputBuffer::ARRAY_DATA_TYPE;
        NdArrayDef definition{{totalSize/NdArrayDef::getDataTypeSize(DATA_TYPE)}, DATA_TYPE};
        Us4OEMBufferArrayDef def{0, definition, parts};
        Us4OEMBufferElement element{0, totalSize, 0};
        builder.add(def);
        builder.add(element);
        return builder.build();
    }

    std::vector<Us4OEMDataTransferRegistrar::ArrayTransfers> createTransfers(const Us4OEMBufferArrayParts &parts) {
        auto oemBuffer = createUs4OEMBuffer(parts);
        auto dst = createDstBuffer(oemBuffer);
        return Us4OEMDataTransferRegistrar::createTransfers(dst.get(), oemBuffer, 0, defaultDescriptor.getMaxTransferSize());
    }

    Us4OEMDescriptor defaultDescriptor = DEFAULT_DESCRIPTOR;
};


TEST_F(Us4OEMDataTransferRegistrarTest, CorrectlyPassesSinglePartAsSingleTransfer) {
    Us4OEMBufferArrayParts parts = {
        Us4OEMBufferArrayPart{0, 4096,0, 14, 4096},
    };
    auto transfers = createTransfers(parts);
    std::vector<Us4OEMDataTransferRegistrar::ArrayTransfers> expected{{
        Transfer{0, 0, 4096, 14}
    }};
    ASSERT_EQ(transfers, expected);
}

TEST_F(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoSingleTransfer) {
    // Given
    Us4OEMBufferArrayParts parts;
    size_t totalSize = defaultDescriptor.getMaxTransferSize() - 64;
    unsigned nSamples = 4096;
    size_t partSize = nSamples*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i, 4096});
    }
    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts, 4096});
    auto transfers = createTransfers(parts);

    std::vector<Us4OEMDataTransferRegistrar::ArrayTransfers> expected{{
        Transfer{0, 0, totalSize, (uint16_t) (nFullParts)}
    }};
    ASSERT_EQ(transfers, expected);
}

TEST_F(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoTwoTransfers) {
    // Given
    Us4OEMBufferArrayParts parts;
    auto maxTransferSize = defaultDescriptor.getMaxTransferSize();
    size_t totalSize = maxTransferSize + 64;
    unsigned nSamples = 4096;
    size_t partSize = nSamples*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i, nSamples});
    }

    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts, nSamples});

    auto transfers = createTransfers(parts);
    // expect:
    std::vector<Us4OEMDataTransferRegistrar::ArrayTransfers> expected{{
            Transfer{0, 0, maxTransferSize, (uint16_t)(nFullParts-1)},
            Transfer{maxTransferSize, maxTransferSize, 64, (uint16_t)(nFullParts)}
    }};
    ASSERT_EQ(transfers, expected);
}

TEST_F(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoThreeTransfers) {
    // Given
    Us4OEMBufferArrayParts parts;
    auto maxTransferSize = defaultDescriptor.getMaxTransferSize();
    size_t totalSize = 2*maxTransferSize + 128;
    unsigned nSamples = 4096;
    size_t partSize = nSamples*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i, nSamples});
    }
    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts, nSamples});

    auto transfers = createTransfers(parts);

    // expect:
    std::vector<Us4OEMDataTransferRegistrar::ArrayTransfers> expected{{
            Transfer{0, 0, maxTransferSize, (uint16_t)((nFullParts-1)/2)},
            Transfer{maxTransferSize, maxTransferSize, maxTransferSize, (uint16_t)(nFullParts-1)},
            Transfer{2*maxTransferSize, 2*maxTransferSize, 128, (uint16_t)(nFullParts)},
    }};
    ASSERT_EQ(transfers, expected);
}
}


// --- equal-sized transfers, which the Ethernet transport requires -------------

TEST_F(Us4OEMDataTransferRegistrarTest, SplitsAnStaElementIntoEqualHalvesRatherThanAFullAndARemainder) {
    // A 192-transmit STA element over the esaote3 adapter: 3 receive firings per transmit, each
    // 3008 samples x 32 channels x 2 B = 188 KiB, so 105.75 MiB against the 64 MiB transfer limit.
    // Filling greedily gave 64 MiB + 41.9 MiB, which the Ethernet receiver refuses ("transfer set
    // must be ... all of the same length"), since the bridge egresses one fixed-size frame per
    // transfer.
    constexpr size_t PART = 3008 * 32 * 2;
    constexpr uint16_t N_PARTS = 192 * 3;
    Us4OEMBufferArrayParts parts;
    for (uint16_t i = 0; i < N_PARTS; ++i) {
        parts.push_back(Us4OEMBufferArrayPart{i * PART, PART, 0, i, 3008});
    }

    auto transfers = createTransfers(parts);

    ASSERT_EQ(transfers.size(), 1);
    ASSERT_EQ(transfers[0].size(), 2);
    const size_t half = (N_PARTS / 2) * PART;
    EXPECT_EQ(transfers[0][0], (Transfer{0, 0, half, (uint16_t)(N_PARTS / 2 - 1)}));
    EXPECT_EQ(transfers[0][1], (Transfer{half, half, half, (uint16_t)(N_PARTS - 1)}));
    EXPECT_LE(half, defaultDescriptor.getMaxTransferSize());
}

TEST_F(Us4OEMDataTransferRegistrarTest, EqualPartsThatDivideEvenlyGiveEquallySizedTransfers) {
    // 128 MiB of 1 MiB parts: two transfers of exactly the limit.
    const size_t maxTransferSize = defaultDescriptor.getMaxTransferSize();
    constexpr size_t PART = 1u << 20;
    const auto nParts = (uint16_t)(2 * maxTransferSize / PART);
    Us4OEMBufferArrayParts parts;
    for (uint16_t i = 0; i < nParts; ++i) { parts.push_back(Us4OEMBufferArrayPart{i * PART, PART, 0, i, 4096}); }

    auto transfers = createTransfers(parts);

    ASSERT_EQ(transfers[0].size(), 2);
    EXPECT_EQ(transfers[0][0].size, maxTransferSize);
    EXPECT_EQ(transfers[0][1].size, maxTransferSize);
}

TEST(SplitIntoEqualTransfersTest, PrefersTheFewestEqualTransfersAndRefusesToSpendTooMany) {
    const size_t MiB = 1u << 20;
    // One transfer when everything fits.
    EXPECT_EQ(splitIntoEqualTransfers(std::vector<size_t>(96, 256 * 1024), 64 * MiB),
              (std::vector<size_t>{96}));
    // Three when two would exceed the limit.
    EXPECT_EQ(splitIntoEqualTransfers(std::vector<size_t>(300, MiB), 128 * MiB),
              (std::vector<size_t>{100, 100, 100}));
    // A prime number of parts could only be cut one-per-transfer: refused, since each transfer
    // costs one of the board's 256 descriptors. The caller then fills greedily.
    EXPECT_TRUE(splitIntoEqualTransfers(std::vector<size_t>(7, 20 * MiB), 64 * MiB).empty());
    // Nothing to split.
    EXPECT_TRUE(splitIntoEqualTransfers({}, 64 * MiB).empty());
    EXPECT_TRUE(splitIntoEqualTransfers(std::vector<size_t>(4, MiB), 0).empty());
}


int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

