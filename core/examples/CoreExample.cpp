#include <iostream>

#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/common/logging.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"

int main() noexcept {
    using namespace ::arrus::ops::us4r;
    try {
        auto loggingMechanism = std::make_shared<::arrus::Logging>();
        std::shared_ptr<std::ostream> ostream{
            std::shared_ptr<std::ostream>(&std::cout, [](std::ostream *) {})};
        loggingMechanism->addTextSink(ostream, ::arrus::LogSeverity::TRACE);

        ::arrus::setLoggerFactory(loggingMechanism);

        auto settings =
            ::arrus::io::readSessionSettings(
                "C:\\Users\\pjarosik\\src\\x-files\\customers\\nanoecho\\nanoecho_magprobe_002.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        ::arrus::BitMask txAperture(192, true);
        ::arrus::BitMask rxAperture(192, true);
        std::vector<float> delays(192, 0.0f);

        Pulse pulse(4e6, 3.5, false);
        ::arrus::Interval<::arrus::uint32> sampleRange{0, 4096};

        TxRxSequence seq(
            {
                {Tx(txAperture, delays, pulse), Rx(rxAperture, sampleRange)}
            },
            1000e-6f, {});

        auto[fcm, buffer] = us4r->upload(seq);
        us4r->start();

        int16_t* data = buffer->tail();
        std::cout << "Waiting for user input" << std::endl;
        std::string str;
        std::cin >> str;
        std::cout << "Done, releasing buffer" << std::endl;

        us4r->stop();

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
