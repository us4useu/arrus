#include <gtest/gtest.h>

#include "Us4RSubsequence.h"
#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/common/tests.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;

TEST(Us4RSubsequenceFactoryTest, HandlesProperlyASingleInputSequence) {
    ::arrus::ops::us4r::TxRxSequence seq{
        {
            TxRx(Tx({true, true, true, true}, {0.0f, 0.0f, 0.0f}, Pulse{1.0e6, 1, false}),
                 Rx({true, true, true, true}, {0, 4096}), 100e-6),
            TxRx(Tx({true, true, true, true}, {0.0f, 0.0f, 0.0f}, Pulse{1.0e6, 1, false}),
                 Rx({true, true, true, true}, {0, 4096}), 100e-6),
        },
        {}};
    // Sequence ID -> OEM -> subsequence
    std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>> oemSequences = {{
        // Single sequence
        TxRxParametersSequence{ // OEM:0
                               {// op 0
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {true, false, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, false, true, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                // op 1
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {true, false, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, false, true, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                }},
                               1,
                               std::nullopt,
                               {},
                               DeviceId(arrus::devices::DeviceType::Probe, 0),
                               DeviceId(arrus::devices::DeviceType::Probe, 0)},
        TxRxParametersSequence{ // OEM:1
                               {// op 0
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, true, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, false, false, true},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                // op 1
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, true, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false},
                                    {false, false, false, true},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                }},
                               1,
                               std::nullopt,
                               {},
                               DeviceId(arrus::devices::DeviceType::Probe, 0),
                               DeviceId(arrus::devices::DeviceType::Probe, 0)},
    }};
    std::vector<LogicalToPhysicalOp> mapping = {{
        {{0, 1}, {2, 3}},// OEM: 0
        {{0, 1}, {2, 3}},// OEM: 1
    }};

    Us4OEMBufferArrayDef arrayDefOEM0{
        0,
        framework::NdArrayDef{{4 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {Us4OEMBufferArrayPart{
             0,
             4096,
             0,
             0,
             4096,
         },
         Us4OEMBufferArrayPart{
             1 * 4096,
             4096,
             0,
             1,
             4096,
         },
         Us4OEMBufferArrayPart{
             2 * 4096,
             4096,
             0,
             2,
             4096,
         },
         Us4OEMBufferArrayPart{
             3 * 4096,
             4096,
             0,
             3,
             4096,
         }}};

    Us4OEMBufferArrayDef arrayDefOEM1{
        0,
        framework::NdArrayDef{{4 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {Us4OEMBufferArrayPart{
             0,
             4096,
             0,
             0,
             4096,
         },
         Us4OEMBufferArrayPart{
             1 * 4096,
             4096,
             0,
             1,
             4096,
         },
         Us4OEMBufferArrayPart{
             2 * 4096,
             4096,
             0,
             2,
             4096,
         },
         Us4OEMBufferArrayPart{
             3 * 4096,
             4096,
             0,
             3,
             4096,
         }}};

    std::vector<Us4OEMBuffer> oemBuffers = {
        Us4OEMBuffer{// OEM:0
                     {Us4OEMBufferElement{0, 4096, 1}, Us4OEMBufferElement{4096, 4096, 3}},
                     {arrayDefOEM0}},
        Us4OEMBuffer{// OEM:1
                     {Us4OEMBufferElement{0, 4096, 1}, Us4OEMBufferElement{4096, 4096, 3}},
                     {arrayDefOEM1}}};

    FrameChannelMappingBuilder fcmBuilder(2, 4);
    // op 0, channel: 0 -> oem: 0, frame: 0, channel: 0
    fcmBuilder.setChannelMapping(0, 0, 0, 0, 0);
    // op 0, channel: 1 -> oem: 1, frame: 0, channel: 0
    fcmBuilder.setChannelMapping(0, 1, 1, 0, 0);
    // op 0, channel: 2 -> oem: 0, frame: 1, channel: 0
    fcmBuilder.setChannelMapping(0, 2, 0, 1, 0);
    // op 0, channel: 3 -> oem: 1, frame: 1, channel: 0
    fcmBuilder.setChannelMapping(0, 3, 1, 1, 0);

    // op 1, channel: 0 -> oem: 0, frame: 2, channel: 0
    fcmBuilder.setChannelMapping(1, 0, 0, 2, 0);
    // op 1, channel: 1 -> oem: 1, frame: 2, channel: 0
    fcmBuilder.setChannelMapping(1, 1, 1, 2, 0);
    // op 1, channel: 2 -> oem: 0, frame: 3, channel: 0
    fcmBuilder.setChannelMapping(1, 2, 0, 3, 0);
    // op 1, channel: 3 -> oem: 1, frame: 3, channel: 0
    fcmBuilder.setChannelMapping(1, 3, 1, 3, 0);

    auto fcm = fcmBuilder.build();
    std::vector<FrameChannelMappingImpl::Handle> fcms;
    fcms.emplace_back(std::move(fcm));

    Us4RSubsequenceFactory factory{{seq}, mapping, oemSequences, oemBuffers, fcms};

    const auto res = factory.get(0, 1, 1, std::nullopt);
    EXPECT_EQ(res.getStart(), 2);
    EXPECT_EQ(res.getEnd(), 3);

    // OEM buffers:
    const auto &resultOEMBuffers = res.getOemBuffers();
    const auto &oem0Buffer = resultOEMBuffers.at(0);
    EXPECT_EQ(oem0Buffer.getElement(0).getSize(), 2 * 4096);
    EXPECT_EQ(oem0Buffer.getElement(0).getAddress(), 2 * 4096);
    EXPECT_EQ(oem0Buffer.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({2 * 4096, 1}));
    EXPECT_EQ(oem0Buffer.getParts(0).at(0).getEntryId(), 2);
    EXPECT_EQ(oem0Buffer.getParts(0).at(1).getEntryId(), 3);
    const auto &oem1Buffer = resultOEMBuffers.at(1);
    EXPECT_EQ(oem1Buffer.getElement(0).getSize(), 2 * 4096);
    EXPECT_EQ(oem1Buffer.getElement(0).getAddress(), 2 * 4096);
    EXPECT_EQ(oem1Buffer.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({2 * 4096, 1}));
    EXPECT_EQ(oem1Buffer.getParts(0).at(0).getEntryId(), 2);
    EXPECT_EQ(oem1Buffer.getParts(0).at(1).getEntryId(), 3);

    // FCM:
    auto outputFCM = res.buildFCM();
    EXPECT_EQ(outputFCM->getNumberOfLogicalFrames(), 1);
    EXPECT_EQ(outputFCM->getNumberOfLogicalChannels(), 4);
    EXPECT_EQ(outputFCM->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
    EXPECT_EQ(outputFCM->getLogical(0, 1), FrameChannelMappingAddress(1, 0, 0));
    EXPECT_EQ(outputFCM->getLogical(0, 2), FrameChannelMappingAddress(0, 1, 0));
    EXPECT_EQ(outputFCM->getLogical(0, 3), FrameChannelMappingAddress(1, 1, 0));
}

TEST(Us4RSubsequenceFactoryTest, HandlesProperlyTwoSequences) {

    // mapping:
    // system channel -> OEM, OEM channel
    // 0 -> (0, 0)
    // 1 -> (1, 0)

    std::vector<::arrus::ops::us4r::TxRxSequence> seqs{
        {// Sequence 0
         {
             TxRx(Tx({true, true}, {0.0f, 0.0f}, Pulse{1.0e6, 1, false}), Rx({true, true}, {0, 4096}), 100e-6),
             TxRx(Tx({true, true}, {1.0e-6f, 2.0e-6f}, Pulse{1.0e6, 1, false}), Rx({true, true}, {0, 4096}), 100e-6),
         },
         {}},
        {// Sequence 1
         {
             TxRx(Tx({false, true}, {3.0e-6f}, Pulse{3.0e6, 1, false}), Rx({true, false}, {0, 4096}), 100e-6),
         },
         {}}};

    // Physical sequences applied on each OEM (translated (manually) based on the above sequence).
    // Sequence ID -> OEM -> subsequence
    std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>> oemSequences = {
        {
            // Sequence 0
            TxRxParametersSequence{// OEM:0
                                   {
                                       // op 0
                                       TxRxParameters{
                                           {true},
                                           {0.0f, 0.0f},
                                           Pulse{1.0e6, 1, false},
                                           {true},
                                           {0, 4096},
                                           1,
                                           100e-6,
                                       },
                                       // op 1
                                       TxRxParameters{
                                           {true},
                                           {1.0e-6f},
                                           Pulse{1.0e6, 1, false},
                                           {true},
                                           {0, 4096},
                                           1,
                                           100e-6,
                                       },
                                   },
                                   1,
                                   std::nullopt,
                                   {},
                                   DeviceId(arrus::devices::DeviceType::Probe, 0),
                                   DeviceId(arrus::devices::DeviceType::Probe, 0)},
            TxRxParametersSequence{ // OEM:1
                                   {// op 0
                                    TxRxParameters{
                                        {true},
                                        {0.0f},
                                        Pulse{1.0e6, 1, false},
                                        {true},
                                        {0, 4096},
                                        1,
                                        100e-6,
                                    },
                                    // op 1
                                    TxRxParameters{
                                        {true},
                                        {2.0e-6f},
                                        Pulse{1.0e6, 1, false},
                                        {true},
                                        {0, 4096},
                                        1,
                                        100e-6,
                                    }},
                                   1,
                                   std::nullopt,
                                   {},
                                   DeviceId(arrus::devices::DeviceType::Probe, 0),
                                   DeviceId(arrus::devices::DeviceType::Probe, 0)},
        },
        {
            // Sequence 1
            TxRxParametersSequence{// OEM:0
                                   {
                                       // op 0
                                       TxRxParameters{
                                           {false},
                                           {0.0f},
                                           Pulse{3.0e6, 1, false},
                                           {true},
                                           {0, 4096},
                                           1,
                                           100e-6,
                                       },
                                   },
                                   1,
                                   std::nullopt,
                                   {},
                                   DeviceId(arrus::devices::DeviceType::Probe, 0),
                                   DeviceId(arrus::devices::DeviceType::Probe, 0)},
            TxRxParametersSequence{// OEM:0
                                   {
                                       // op 0
                                       TxRxParameters{
                                           {true},
                                           {0.0f},
                                           Pulse{3.0e6, 1, false},
                                           {false},
                                           {0, 4096},
                                           1,
                                           100e-6,
                                       },
                                   },
                                   1,
                                   std::nullopt,
                                   {},
                                   DeviceId(arrus::devices::DeviceType::Probe, 0),
                                   DeviceId(arrus::devices::DeviceType::Probe, 0)},
        }};
    // Logical -> physical op ranges.
    // e.g. {{0, 1}, {2, 2}} means that the first logical op was translated to two physical ops 0 and 1,
    // second logical op was translated to a single physical op 2.
    std::vector<LogicalToPhysicalOp> mapping = {{
                                                    // sequence 0
                                                    {0, 0},
                                                    {1, 1},
                                                },
                                                {
                                                    // sequence 1
                                                    {0, 0},
                                                }};

    // Output array: sequence 0.
    Us4OEMBufferArrayDef arrayDefOEM0Seq0{
        // OEM 0
        0,
        framework::NdArrayDef{{2 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {
            Us4OEMBufferArrayPart{
                0,
                4096,
                0,
                0,
                4096,
            },
            Us4OEMBufferArrayPart{
                1 * 4096,
                4096,
                0,
                1,
                4096,
            },
        }};

    Us4OEMBufferArrayDef arrayDefOEM1Seq0{
        // OEM 1
        0,
        framework::NdArrayDef{{2 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {Us4OEMBufferArrayPart{
             0,
             4096,
             0,
             0,
             4096,
         },
         Us4OEMBufferArrayPart{
             1 * 4096,
             4096,
             0,
             1,
             4096,
         }}};

    // Output array: sequence 1
    Us4OEMBufferArrayDef arrayDefOEM0Seq1{ // OEM 0
        0,
        framework::NdArrayDef{{1 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {Us4OEMBufferArrayPart{
            0,
            4096,
            0,
            2,
            4096,
        }}};

    Us4OEMBufferArrayDef arrayDefOEM1Seq1{ // OEM 1
        0,
        framework::NdArrayDef{{0 * 4096, 1}, arrus::framework::NdArrayDef::DataType::INT16},
        {Us4OEMBufferArrayPart{
            0,
            0,
            0,
            2,
            0,
        }}};

    std::vector<Us4OEMBuffer> oemBuffers = {
        Us4OEMBuffer{// OEM:0
                     {Us4OEMBufferElement{0, 4096, 0}, Us4OEMBufferElement{4096, 4096, 1}},
                     {arrayDefOEM0Seq0, arrayDefOEM0Seq1}
        },
        Us4OEMBuffer{// OEM:1
                     {Us4OEMBufferElement{0, 4096, 0}, Us4OEMBufferElement{4096, 4096, 1}},
                     {arrayDefOEM1Seq0, arrayDefOEM1Seq1}
        }
    };

    // Sequence 0
    FrameChannelMappingBuilder fcmBuilder(2, 2);
    // op 0, channel: 0 -> oem: 0, frame: 0, channel: 0
    fcmBuilder.setChannelMapping(0, 0, 0, 0, 0);
    // op 0, channel: 1 -> oem: 1, frame: 0, channel: 0
    fcmBuilder.setChannelMapping(0, 1, 1, 0, 0);

    // op 1, channel: 0 -> oem: 0, frame: 1, channel: 0
    fcmBuilder.setChannelMapping(1, 0, 0, 1, 0);
    // op 1, channel: 1 -> oem: 1, frame: 1, channel: 0
    fcmBuilder.setChannelMapping(1, 1, 1, 1, 0);

    // Sequence 1
    FrameChannelMappingBuilder fcmBuilder2(1, 1);
    // op 0, channel: 0 -> oem: 0, frame: 0, channel: 0
    fcmBuilder2.setChannelMapping(0, 0, 0, 0, 0);

    std::vector<FrameChannelMappingImpl::Handle> fcms;
    fcms.emplace_back(std::move(fcmBuilder.build()));
    fcms.emplace_back(std::move(fcmBuilder2.build()));

    Us4RSubsequenceFactory factory{seqs, mapping, oemSequences, oemBuffers, fcms};

    // Select sequence 0, only the first TX/RX
    const auto res = factory.get(0, 0, 0, std::nullopt);
    EXPECT_EQ(res.getStart(), 0);
    EXPECT_EQ(res.getEnd(), 0);

    // OEM buffers:
    const auto &resultOEMBuffers = res.getOemBuffers();
    const auto &oem0Buffer = resultOEMBuffers.at(0);
    EXPECT_EQ(oem0Buffer.getElement(0).getSize(), 4096);
    EXPECT_EQ(oem0Buffer.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem0Buffer.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({1 * 4096, 1}));
    EXPECT_EQ(oem0Buffer.getParts(0).at(0).getEntryId(), 0);
    const auto &oem1Buffer = resultOEMBuffers.at(1);
    EXPECT_EQ(oem1Buffer.getElement(0).getSize(), 4096);
    EXPECT_EQ(oem1Buffer.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem1Buffer.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({1 * 4096, 1}));
    EXPECT_EQ(oem1Buffer.getParts(0).at(0).getEntryId(), 0);

    // FCM:
    auto outputFCM = res.buildFCM();
    EXPECT_EQ(outputFCM->getNumberOfLogicalFrames(), 1);
    EXPECT_EQ(outputFCM->getNumberOfLogicalChannels(), 2);
    EXPECT_EQ(outputFCM->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
    EXPECT_EQ(outputFCM->getLogical(0, 1), FrameChannelMappingAddress(1, 0, 0));
    // Select sequence 1.

    const auto res1 = factory.get(1, 0, 0, std::nullopt);
    EXPECT_EQ(res1.getStart(), 2);
    EXPECT_EQ(res1.getEnd(), 2);

    // OEM buffers:
    const auto &resultOEMBuffers1 = res1.getOemBuffers();
    const auto &oem0Buffer1 = resultOEMBuffers1.at(0);
    EXPECT_EQ(oem0Buffer1.getElement(0).getSize(), 4096);
    EXPECT_EQ(oem0Buffer1.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem0Buffer1.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({1 * 4096, 1}));
    EXPECT_EQ(oem0Buffer1.getParts(0).at(0).getEntryId(), 2);
    const auto &oem1Buffer1 = resultOEMBuffers1.at(1);
    EXPECT_EQ(oem1Buffer1.getElement(0).getSize(), 0);
    EXPECT_EQ(oem1Buffer1.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem1Buffer1.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({0, 1}));
    EXPECT_EQ(oem1Buffer1.getParts(0).at(0).getEntryId(), 2);

    // FCM:
    auto outputFCM1 = res1.buildFCM();
    EXPECT_EQ(outputFCM1->getNumberOfLogicalFrames(), 1);
    EXPECT_EQ(outputFCM1->getNumberOfLogicalChannels(), 1);
    EXPECT_EQ(outputFCM1->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
}

}// namespace

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}