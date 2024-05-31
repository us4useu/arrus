#include <gtest/gtest.h>

#include "arrus/core/common/logging.h"
#include "TxTimeoutRegister.h"
#include "arrus/core/devices/us4r/us4oem/tests/CommonSettings.h"
#include "arrus/core/common/tests.h"
#include "arrus/common/format.h"


namespace {

using namespace arrus;
using namespace arrus::devices;
using namespace arrus::devices::us4r;
using namespace arrus::ops::us4r;

TEST(TxTimeoutRegisterFactoryTest, Calculate) {
    constexpr ChannelIdx nChannels = 32;
    std::vector<float> delays0(nChannels, 0.0f);
    std::vector<float> delays1(nChannels, 0.0f);
    BitMask txAperture(nChannels, true);
    BitMask rxAperture(nChannels, true);

    ops::us4r::Pulse pulse0{10.0e6f, 5000.0f, false}; // 500 us pulse
    ops::us4r::Pulse pulse1{3.0e6f, 3.0f, true}; // 1 us pulse

    delays0.at(21) = 10e-6f;
    delays1.at(7) = 1e-6f;

    std::vector<TxRx> txrxs = {
        // Linearly increasing TX delays.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse0
            )
        ).getTxRx(),
        // All TX delays the same.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays1,
                x.pulse = pulse1
            )
        ).getTxRx(),
        // Partial TX aperture.
        ARRUS_STRUCT_INIT_LIST(
            TestTxRxParams,
            (
                x.txAperture = txAperture,
                x.rxAperture = rxAperture,
                x.txDelays = delays0,
                x.pulse = pulse1
            )
        ).getTxRx()
    };

    TxRxSequence sequence{txrxs, {}};

    TxTimeoutRegisterFactory factory{4, [](float frequency) {return frequency;}};
    auto reg = factory.createFor({sequence});

    std::cout << "Timeouts: " << std::endl;
    std::cout << ::arrus::toString(reg.getTimeouts()) << std::endl;
}
}


int main(int argc, char **argv) {
    ARRUS_INIT_TEST_LOG(arrus::Logging);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}