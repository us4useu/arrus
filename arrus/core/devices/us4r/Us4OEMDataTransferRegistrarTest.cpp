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
        Us4OEMBufferArrayPart{0, 4096,0, 14},
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
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i});
    }
    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts });
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
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i});
    }
    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts});

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
        parts.push_back(Us4OEMBufferArrayPart{i*partSize, partSize, 0, (uint16_t)i});
    }
    parts.push_back(Us4OEMBufferArrayPart{nFullParts*partSize, totalSize-nFullParts*partSize, 0, (uint16_t)nFullParts});

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


int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

