#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/devices/us4r/probeadapter/Us4OEMDataTransferRegistrar.h"

namespace {

using ::arrus::devices::Us4OEMDataTransferRegistrar;
using ::arrus::devices::Us4OEMBufferElementPart;
using ::arrus::devices::Transfer;
using ::arrus::devices::Us4OEMImpl;

TEST(Us4OEMDataTransferRegistrarTest, CorrectlyPassesSinglePartAsSingleTransfer) {
    std::vector<Us4OEMBufferElementPart> parts{
        Us4OEMBufferElementPart{0, 4096, 14},
    };
    auto transfers = Us4OEMDataTransferRegistrar::groupPartsIntoTransfers(parts);
    std::vector<Transfer> expected{
        Transfer{0, 4096, 14}
    };
    ASSERT_EQ(transfers, expected);
}

TEST(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoSingleTransfer) {
    // Given
    std::vector<Us4OEMBufferElementPart> parts;
    size_t totalSize = ::arrus::devices::Us4OEMImpl::MAX_TRANSFER_SIZE - 64;
    size_t partSize = 4096*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {parts.push_back(Us4OEMBufferElementPart{i*partSize, partSize, (uint16_t)i});}
    parts.push_back(Us4OEMBufferElementPart{nFullParts*partSize, totalSize-nFullParts*partSize, (uint16_t)nFullParts});

    auto transfers = Us4OEMDataTransferRegistrar::groupPartsIntoTransfers(parts);
    std::vector<Transfer> expected{
            Transfer{0, totalSize, (uint16_t)(nFullParts)}
    };
    ASSERT_EQ(transfers, expected);
}

TEST(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoTwoTransfers) {
    // Given
    std::vector<Us4OEMBufferElementPart> parts;
    size_t totalSize = Us4OEMImpl::MAX_TRANSFER_SIZE + 64;
    size_t partSize = 4096*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {parts.push_back(Us4OEMBufferElementPart{i*partSize, partSize, (uint16_t)i});}
    parts.push_back(Us4OEMBufferElementPart{nFullParts*partSize, totalSize-nFullParts*partSize, (uint16_t)nFullParts});

    auto transfers = Us4OEMDataTransferRegistrar::groupPartsIntoTransfers(parts);

    // expect:
    std::vector<Transfer> expected{
            Transfer{0, Us4OEMImpl::MAX_TRANSFER_SIZE, (uint16_t)(nFullParts-1)},
            Transfer{Us4OEMImpl::MAX_TRANSFER_SIZE, 64, (uint16_t)(nFullParts)}
    };
    ASSERT_EQ(transfers, expected);
}

TEST(Us4OEMDataTransferRegistrarTest, CorrectlyGroupsMultiplePartsIntoThreeTransfers) {
    // Given
    std::vector<Us4OEMBufferElementPart> parts;
    size_t totalSize = 2*Us4OEMImpl::MAX_TRANSFER_SIZE + 128;
    size_t partSize = 4096*128*2;
    size_t nFullParts = totalSize / partSize;
    for(int i = 0; i < nFullParts; ++i) {parts.push_back(Us4OEMBufferElementPart{i*partSize, partSize, (uint16_t)i});}
    parts.push_back(Us4OEMBufferElementPart{nFullParts*partSize, totalSize-nFullParts*partSize, (uint16_t)nFullParts});

    auto transfers = Us4OEMDataTransferRegistrar::groupPartsIntoTransfers(parts);

    // expect:
    std::vector<Transfer> expected{
            Transfer{0, Us4OEMImpl::MAX_TRANSFER_SIZE, (uint16_t)((nFullParts-1)/2)},
            Transfer{Us4OEMImpl::MAX_TRANSFER_SIZE, Us4OEMImpl::MAX_TRANSFER_SIZE, (uint16_t)(nFullParts-1)},
            Transfer{2*Us4OEMImpl::MAX_TRANSFER_SIZE, 128, (uint16_t)(nFullParts)},
    };
    ASSERT_EQ(transfers, expected);
}

}


int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

