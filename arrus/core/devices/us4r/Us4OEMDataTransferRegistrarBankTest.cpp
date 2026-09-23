#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <map>
#include <tuple>

#include "Us4OEMDataTransferRegistrar.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace ::arrus;
using namespace ::arrus::devices;
using namespace ::arrus::framework;
using ::testing::_;
using ::testing::Truly;

/**
 * The data transfers of a sub-sequence programmed in the second bank of the sequencer table
 * (the sequencer double-buffering): verifies exactly what is programmed in the us4OEM.
 */
class Us4OEMDataTransferRegistrarBankTest : public ::testing::Test {
protected:
    static constexpr size_t PART_SIZE = 64*1024;
    static constexpr uint16 N_FIRINGS = 4;// per element
    static constexpr uint16 BANK_OFFSET = 2048;
    static constexpr size_t ID_OFFSET = 64;

    void SetUp() override {
        auto ius4oem = std::make_unique<::testing::NiceMock<MockIUs4OEM>>();
        mock = ius4oem.get();
        ON_CALL(*mock, GetOemVersion).WillByDefault(testing::Return(2));
        auto rxSettings = RxSettingsBuilder()
                              .setActiveTermination(std::nullopt).setPgaGain(30).setLnaGain(24).setTgcSamples({})
                              .setLpfCutoff(15'000'000).setDtgcAttenuation(std::nullopt)
                              .setApplyTgcCharacteristic(true).build();
        us4oem = std::make_unique<Us4OEMImpl>(DeviceId(DeviceType::Us4OEM, 0), std::move(ius4oem),
                                              getRange<uint8>(0, 128), rxSettings,
                                              Us4OEMSettings::ReprogrammingMode::SEQUENTIAL, DEFAULT_DESCRIPTOR,
                                              false, false);
    }

    /** The full sequence: 2 elements, N_FIRINGS parts each (one part per firing). */
    Us4OEMBuffer createFullBuffer() {
        Us4OEMBufferBuilder builder;
        Us4OEMBufferArrayParts parts;
        for (uint16 f = 0; f < N_FIRINGS; ++f) {
            parts.emplace_back(f*PART_SIZE, PART_SIZE, 0, f, 1024);
        }
        const auto dataType = Us4ROutputBuffer::ARRAY_DATA_TYPE;
        const size_t elementSize = N_FIRINGS*PART_SIZE;
        NdArrayDef definition{{elementSize/NdArrayDef::getDataTypeSize(dataType)/32, 32}, dataType};
        builder.add(Us4OEMBufferArrayDef{0, definition, parts});
        builder.add(Us4OEMBufferElement{0, elementSize, N_FIRINGS-1});
        builder.add(Us4OEMBufferElement{elementSize, elementSize, 2*N_FIRINGS-1});
        return builder.build();
    }

    /** Sub-sequence: firings 0, 2, 3 (i.e. two transfers per element: [0], [2, 3]). */
    Us4OEMBuffer createSubsequenceBuffer(const Us4OEMBuffer &full) {
        const auto &fullParts = full.getParts(0);
        Us4OEMBufferArrayParts parts = {fullParts.at(0), fullParts.at(2), fullParts.at(3)};
        const auto dataType = Us4ROutputBuffer::ARRAY_DATA_TYPE;
        const size_t size = 3*PART_SIZE;
        NdArrayDef definition{{size/NdArrayDef::getDataTypeSize(dataType)/32, 32}, dataType};
        Us4OEMBufferBuilder builder;
        builder.add(Us4OEMBufferArrayDef{0, definition, parts});
        for (const auto &element : full.getElements()) {
            builder.add(Us4OEMBufferElement{element.getAddress(), size, element.getGlobalFiring(), 3});
        }
        return builder.build();
    }

    Us4ROutputBuffer::SharedHandle createDstBuffer(const Us4OEMBuffer &buffer) {
        Us4ROutputBufferBuilder builder;
        return builder.setStopOnOverflow(false).setNumberOfElements(2).setLayoutTo({buffer}).setUseP2pDma(false).build();
    }

    ::testing::NiceMock<MockIUs4OEM> *mock{nullptr};
    Us4OEMImpl::Handle us4oem;
};

TEST_F(Us4OEMDataTransferRegistrarBankTest, ProgramsTheTransfersOfTheSecondBank) {
    auto full = createFullBuffer();
    auto src = createSubsequenceBuffer(full);
    auto dst = createDstBuffer(src);

    // Page locks and unlocks must be balanced: (dst, size, src) -> lock count - unlock count
    std::map<std::tuple<unsigned char *, size_t, size_t>, int> locks;
    ON_CALL(*mock, PrepareHostBuffer(_, _, _, _))
        .WillByDefault([&](unsigned char *d, size_t size, size_t s, bool) { ++locks[{d, size, s}]; });
    ON_CALL(*mock, ReleaseTransferRxBufferToHost(_, _, _))
        .WillByDefault([&](unsigned char *d, size_t size, size_t s) { --locks[{d, size, s}]; });
    std::vector<size_t> preparedIds;
    ON_CALL(*mock, PrepareTransferRXBufferToHost(_, _, _, _, _))
        .WillByDefault([&](size_t idx, unsigned char *, size_t, size_t, bool) { preparedIds.push_back(idx); });
    std::vector<std::pair<size_t, size_t>> scheduled;// firing, idx
    EXPECT_CALL(*mock, ScheduleTransferRXBufferToHost(_, _, Truly([](const std::function<void()> &f) { return !f; })))
        .Times(4)
        .WillRepeatedly([&](size_t firing, size_t idx, const std::function<void()> &) {
            scheduled.emplace_back(firing, idx);
        });
    // The us4OEM IRQ callbacks must not be touched (the device is running).
    EXPECT_CALL(*mock, RegisterCallback(_, _)).Times(0);

    Us4OEMDataTransferRegistrar registrar(dst.get(), src, us4oem.get(), DEFAULT_DESCRIPTOR.getMaxTransferSize(),
                                          BANK_OFFSET, ID_OFFSET, 64);
    registrar.registerTransfers();

    EXPECT_EQ(registrar.getStrategy(), 0);
    EXPECT_EQ(registrar.getNumberOfTransfersPerElement(), 2);
    EXPECT_EQ(registrar.getCallbacks().size(), 4);
    EXPECT_EQ(preparedIds, (std::vector<size_t>{64, 65, 66, 67}));
    // Element 0: firings [0] and [2, 3]; element 1 starts at the firing 4. All in the second bank.
    std::vector<std::pair<size_t, size_t>> expectedScheduled = {
        {BANK_OFFSET + 0, 64}, {BANK_OFFSET + 3, 65}, {BANK_OFFSET + 4, 66}, {BANK_OFFSET + 7, 67}};
    EXPECT_EQ(scheduled, expectedScheduled);
    EXPECT_EQ(locks.size(), 4);

    // Unregistering: releases exactly what was locked, clears the requests of the second bank only.
    std::vector<size_t> cleared;
    EXPECT_CALL(*mock, ClearTransferRXBufferToHost(_))
        .Times(4).WillRepeatedly([&](size_t firing) { cleared.push_back(firing); });
    registrar.unregisterTransfers(true);
    std::sort(std::begin(cleared), std::end(cleared));
    EXPECT_EQ(cleared, (std::vector<size_t>{BANK_OFFSET + 0, BANK_OFFSET + 3, BANK_OFFSET + 4, BANK_OFFSET + 7}));
    for (const auto &[key, count] : locks) {
        EXPECT_EQ(count, 0) << "Unbalanced page lock for the transfer of size " << std::get<1>(key);
    }
}

}// namespace

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
