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
            TxRx(Tx({true, true, true, true}, {0.0f, 0.0f, 0.0f}, Pulse{1.0e6, 1, false}.toWaveform()),
                 Rx({true, true, true, true}, {0, 4096}), 100e-6),
            TxRx(Tx({true, true, true, true}, {0.0f, 0.0f, 0.0f}, Pulse{1.0e6, 1, false}.toWaveform()),
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
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {true, false, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {false, false, true, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                // op 1
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {true, false, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
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
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {false, true, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {false, false, false, true},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                // op 1
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
                                    {false, true, false, false},
                                    {0, 4096},
                                    1,
                                    100e-6,
                                },
                                TxRxParameters{
                                    {true, true, true, true},
                                    {0.0f, 0.0f, 0.0f, 0.0f},
                                    Pulse{1.0e6, 1, false}.toWaveform(),
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
        {{0, 2}, {2, 4}},// OEM: 0
        {{0, 2}, {2, 4}},// OEM: 1
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

    const auto res = factory.get(0, 1, 2, std::nullopt);
    const auto resultOEMBuffers = factory.recreateOEMBuffers({res.getArrayDefs()});
    EXPECT_EQ(res.getStart(), 2);
    EXPECT_EQ(res.getEnd(), 4);

    // OEM buffers:
    const auto &oem0Buffer = resultOEMBuffers.at(0);
    EXPECT_EQ(oem0Buffer.getElement(0).getSize(), 2 * 4096);
    EXPECT_EQ(oem0Buffer.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem0Buffer.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({2 * 4096, 1}));
    EXPECT_EQ(oem0Buffer.getParts(0).at(0).getEntryId(), 2);
    EXPECT_EQ(oem0Buffer.getParts(0).at(1).getEntryId(), 3);
    const auto &oem1Buffer = resultOEMBuffers.at(1);
    EXPECT_EQ(oem1Buffer.getElement(0).getSize(), 2 * 4096);
    EXPECT_EQ(oem1Buffer.getElement(0).getAddress(), 0);
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
             TxRx(Tx({true, true}, {0.0f, 0.0f}, Pulse{1.0e6, 1, false}.toWaveform()), Rx({true, true}, {0, 4096}), 100e-6),
             TxRx(Tx({true, true}, {1.0e-6f, 2.0e-6f}, Pulse{1.0e6, 1, false}.toWaveform()), Rx({true, true}, {0, 4096}), 100e-6),
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
                                           Pulse{1.0e6, 1, false}.toWaveform(),
                                           {true},
                                           {0, 4096},
                                           1,
                                           100e-6,
                                       },
                                       // op 1
                                       TxRxParameters{
                                           {true},
                                           {1.0e-6f},
                                           Pulse{1.0e6, 1, false}.toWaveform(),
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
                                        Pulse{1.0e6, 1, false}.toWaveform(),
                                        {true},
                                        {0, 4096},
                                        1,
                                        100e-6,
                                    },
                                    // op 1
                                    TxRxParameters{
                                        {true},
                                        {2.0e-6f},
                                        Pulse{1.0e6, 1, false}.toWaveform(),
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
                                           Pulse{3.0e6, 1, false}.toWaveform(),
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
                                           Pulse{3.0e6, 1, false}.toWaveform(),
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
                                                    {0, 1},
                                                    {1, 2},
                                                },
                                                {
                                                    // sequence 1
                                                    {0, 1},
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
    const auto res = factory.get(0, 0, 1, std::nullopt);
    const auto resultOEMBuffers = factory.recreateOEMBuffers({res.getArrayDefs()});
    EXPECT_EQ(res.getStart(), 0);
    EXPECT_EQ(res.getEnd(), 1);

    // OEM buffers:
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

    const auto res1 = factory.get(1, 0, 1, std::nullopt);
    const auto resultOEMBuffers1 = factory.recreateOEMBuffers({res1.getArrayDefs()});
    EXPECT_EQ(res1.getStart(), 2);
    EXPECT_EQ(res1.getEnd(), 3);

    // OEM buffers:
    const auto &oem0Buffer1 = resultOEMBuffers1.at(0);
    EXPECT_EQ(oem0Buffer1.getElement(0).getSize(), 4096);
    EXPECT_EQ(oem0Buffer1.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem0Buffer1.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({1 * 4096, 1}));
    EXPECT_EQ(oem0Buffer1.getParts(0).at(0).getEntryId(), 2);
    const auto &oem1Buffer1 = resultOEMBuffers1.at(1);
    EXPECT_EQ(oem1Buffer1.getElement(0).getSize(), 0);
    EXPECT_EQ(oem1Buffer1.getElement(0).getAddress(), 0);
    EXPECT_EQ(oem1Buffer1.getArrayDef(0).getDefinition().getShape(), Tuple<size_t>({0, 1}));
    EXPECT_TRUE(oem1Buffer1.getParts(0).empty());

    // FCM:
    auto outputFCM1 = res1.buildFCM();
    EXPECT_EQ(outputFCM1->getNumberOfLogicalFrames(), 1);
    EXPECT_EQ(outputFCM1->getNumberOfLogicalChannels(), 1);
    EXPECT_EQ(outputFCM1->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
}


/**
 * A sequence of 4 TX/RXs (a single physical TX/RX each), acquired by 2 OEMs:
 * - OEM:0 acquires data for each TX/RX (frames 0, 1, 2, 3),
 * - OEM:1 acquires data only for the TX/RXs 1 and 3 (frames 0, 1).
 */
class Us4RSubsequenceFactoryNonConsecutiveTest : public ::testing::Test {
public:
    static constexpr unsigned N_SAMPLES = 4096;
    static constexpr size_t FRAME_SIZE = N_SAMPLES*sizeof(int16_t);

    static TxRxParameters op(bool isRx) {
        return TxRxParameters{
            {true}, {0.0f}, Pulse{1.0e6, 1, false}.toWaveform(), {isRx}, {0, (int)N_SAMPLES}, 1, 100e-6,
        };
    }

    static TxRxParametersSequence oemSequence(const std::vector<bool> &isRx) {
        std::vector<TxRxParameters> ops;
        for(const auto rx: isRx) {
            ops.push_back(op(rx));
        }
        return TxRxParametersSequence{
            ops, 1, std::nullopt, {},
            DeviceId(arrus::devices::DeviceType::Probe, 0),
            DeviceId(arrus::devices::DeviceType::Probe, 0)};
    }

    /** Creates the array definition: a single part for each op, the empty parts for the ops with no RX. */
    static Us4OEMBufferArrayDef arrayDef(const std::vector<bool> &isRx) {
        Us4OEMBufferArrayParts parts;
        size_t address = 0, nFrames = 0;
        for(uint16_t entry = 0; entry < isRx.size(); ++entry) {
            if(isRx.at(entry)) {
                parts.emplace_back(address, FRAME_SIZE, 0, entry, N_SAMPLES);
                address += FRAME_SIZE;
                ++nFrames;
            }
            else {
                parts.emplace_back(address, 0, 0, entry, 0);
            }
        }
        return Us4OEMBufferArrayDef{
            0, framework::NdArrayDef{{nFrames*N_SAMPLES, 1}, arrus::framework::NdArrayDef::DataType::INT16}, parts};
    }

    void SetUp() override {
        std::vector<TxRx> txrxs;
        for(int i = 0; i < 4; ++i) {
            txrxs.emplace_back(Tx({true, true}, {0.0f, 0.0f}, Pulse{1.0e6, 1, false}.toWaveform()),
                               Rx({true, true}, {0, (int)N_SAMPLES}), 100e-6);
        }
        sequences = {TxRxSequence{txrxs, {}}};
        oemSequences = {{oemSequence(oem0Rx), oemSequence(oem1Rx)}};
        mapping = {{{0, 1}, {1, 2}, {2, 3}, {3, 4}}};
        oemBuffers = {
            Us4OEMBuffer{{Us4OEMBufferElement{0, 4*FRAME_SIZE, 3}}, {arrayDef(oem0Rx)}},
            Us4OEMBuffer{{Us4OEMBufferElement{0, 2*FRAME_SIZE, 3}}, {arrayDef(oem1Rx)}},
        };
        // logical (op, channel) -> physical (oem, frame, channel)
        FrameChannelMappingBuilder fcmBuilder(4, 2);
        for(uint16_t op = 0; op < 4; ++op) {
            fcmBuilder.setChannelMapping(op, 0, 0, op, 0);
            if(oem1Rx.at(op)) {
                fcmBuilder.setChannelMapping(op, 1, 1, uint16_t(op/2), 0);
            } // otherwise: the channel is unavailable
        }
        fcms.emplace_back(fcmBuilder.build());
    }

    Us4RSubsequenceFactory getFactory() {
        return Us4RSubsequenceFactory{sequences, mapping, oemSequences, oemBuffers, fcms};
    }

    const std::vector<bool> oem0Rx = {true, true, true, true};
    const std::vector<bool> oem1Rx = {false, true, false, true};
    std::vector<TxRxSequence> sequences;
    std::vector<std::vector<TxRxParametersSequence>> oemSequences;
    std::vector<LogicalToPhysicalOp> mapping;
    std::vector<Us4OEMBuffer> oemBuffers;
    std::vector<FrameChannelMappingImpl::Handle> fcms;
};

TEST_F(Us4RSubsequenceFactoryNonConsecutiveTest, SelectsNonConsecutiveTxRxs) {
    auto factory = getFactory();
    // TX/RXs 0 and 2: only the OEM:0 acquires any data.
    const auto res = factory.get(0, std::vector<uint16_t>{0, 2}, std::nullopt);
    EXPECT_EQ(res.getEntries(), (std::vector<uint16_t>{0, 2}));

    const auto buffers = factory.recreateOEMBuffers({res.getArrayDefs()});
    const auto &oem0 = buffers.at(0);
    EXPECT_EQ(oem0.getElement(0).getSize(), 2*FRAME_SIZE);
    EXPECT_EQ(oem0.getParts(0).size(), 2);
    EXPECT_EQ(oem0.getParts(0).at(0).getEntryId(), 0);
    EXPECT_EQ(oem0.getParts(0).at(1).getEntryId(), 2);
    // The source addresses should be kept (they are relative to the beginning of the FULL buffer element).
    EXPECT_EQ(oem0.getParts(0).at(1).getAddress(), 2*FRAME_SIZE);
    // OEM:1 does not acquire any data for the TX/RXs 0 and 2.
    const auto &oem1 = buffers.at(1);
    EXPECT_EQ(oem1.getElement(0).getSize(), 0);

    auto fcm = res.buildFCM();
    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 2);
    EXPECT_EQ(fcm->getNumberOfFrames(0), 2);
    EXPECT_EQ(fcm->getNumberOfFrames(1), 0);
    // The OEM:0 frames 0, 2 become the frames 0, 1.
    EXPECT_EQ(fcm->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
    EXPECT_EQ(fcm->getLogical(1, 0), FrameChannelMappingAddress(0, 1, 0));
    // The channel 1 is not available for the TX/RXs 0 and 2.
    EXPECT_TRUE(FrameChannelMapping::isChannelUnavailable(fcm->getLogical(0, 1).getChannel()));
    EXPECT_TRUE(FrameChannelMapping::isChannelUnavailable(fcm->getLogical(1, 1).getChannel()));
}

TEST_F(Us4RSubsequenceFactoryNonConsecutiveTest, RenumbersFramesOfAllOEMs) {
    auto factory = getFactory();
    // TX/RXs 1 and 3: both OEMs acquire data.
    const auto res = factory.get(0, std::vector<uint16_t>{1, 3}, std::nullopt);
    EXPECT_EQ(res.getEntries(), (std::vector<uint16_t>{1, 3}));

    const auto buffers = factory.recreateOEMBuffers({res.getArrayDefs()});
    EXPECT_EQ(buffers.at(0).getElement(0).getSize(), 2*FRAME_SIZE);
    EXPECT_EQ(buffers.at(1).getElement(0).getSize(), 2*FRAME_SIZE);
    // The OEM:1 frames are stored one after another in the OEM memory (the ops 0 and 2 acquire nothing).
    EXPECT_EQ(buffers.at(1).getParts(0).at(0).getAddress(), 0);
    EXPECT_EQ(buffers.at(1).getParts(0).at(1).getAddress(), FRAME_SIZE);

    auto fcm = res.buildFCM();
    EXPECT_EQ(fcm->getNumberOfLogicalFrames(), 2);
    EXPECT_EQ(fcm->getNumberOfFrames(0), 2);
    EXPECT_EQ(fcm->getNumberOfFrames(1), 2);
    // OEM:0 frames 1, 3 -> 0, 1
    EXPECT_EQ(fcm->getLogical(0, 0), FrameChannelMappingAddress(0, 0, 0));
    EXPECT_EQ(fcm->getLogical(1, 0), FrameChannelMappingAddress(0, 1, 0));
    // OEM:1 frames 0, 1 -> 0, 1 (unchanged)
    EXPECT_EQ(fcm->getLogical(0, 1), FrameChannelMappingAddress(1, 0, 0));
    EXPECT_EQ(fcm->getLogical(1, 1), FrameChannelMappingAddress(1, 1, 0));
}

TEST_F(Us4RSubsequenceFactoryNonConsecutiveTest, SetsSriUsingTheSelectedTxRxsOnly) {
    auto factory = getFactory();
    // 2 TX/RXs, 100 us each: the last PRI should be extended by 1000-200 = 800 us.
    const auto res = factory.get(0, std::vector<uint16_t>{0, 3}, 1000e-6f);
    EXPECT_EQ(res.getTimeToNextTrigger(), 900);
    // Without SRI: the PRI of the last TX/RX.
    const auto res2 = factory.get(0, std::vector<uint16_t>{0, 3}, std::nullopt);
    EXPECT_EQ(res2.getTimeToNextTrigger(), 100);
}

TEST_F(Us4RSubsequenceFactoryNonConsecutiveTest, TurnsOffTheSequenceForAnEmptyListOfTxRxs) {
    auto factory = getFactory();
    const auto res = factory.get(0, std::vector<uint16_t>{}, std::nullopt);
    EXPECT_TRUE(res.empty());
    EXPECT_TRUE(res.getEntries().empty());
}

TEST_F(Us4RSubsequenceFactoryNonConsecutiveTest, RejectsInvalidListOfTxRxs) {
    auto factory = getFactory();
    // TX/RX outside of the sequence.
    EXPECT_THROW(factory.get(0, std::vector<uint16_t>{0, 4}, std::nullopt), IllegalArgumentException);
    // Not sorted.
    EXPECT_THROW(factory.get(0, std::vector<uint16_t>{2, 1}, std::nullopt), IllegalArgumentException);
    // Repeated.
    EXPECT_THROW(factory.get(0, std::vector<uint16_t>{1, 1}, std::nullopt), IllegalArgumentException);
}

}// namespace

int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(catch_exceptions) = false;
    return RUN_ALL_TESTS();
}