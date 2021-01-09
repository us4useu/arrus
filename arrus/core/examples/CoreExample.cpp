#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>

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
        std::shared_ptr<std::ostream> logFileStream =
            // append to the end of the file
            std::make_shared<std::ofstream>(R"(C:\Users\pjarosik\cpplog.txt)", std::ios_base::app);
        loggingMechanism->addTextSink(logFileStream, arrus::LogSeverity::TRACE);

        ::arrus::setLoggerFactory(loggingMechanism);

        auto settings =
            ::arrus::io::readSessionSettings(
                R"(C:\Users\pjarosik\src\x-files\customers\nanoecho\nanoecho_magprobe_002.prototxt)");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        ::arrus::BitMask txAperture(192, true);
        ::arrus::BitMask rxAperture(192, true);
        std::vector<float> delays(192, 0.0f);

        Pulse pulse(4e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};

        std::vector<TxRx> txrxs;

        for(int i = 0; i < 175; ++i) {
            arrus::BitMask aperture(192, false);

            auto origin = i-32;

            unsigned short leftPadding = 0, rightPadding = 0;
            for(int j = 0; j < 64; ++j) {
                auto idx = origin + j;
                aperture[std::min(std::max(idx, 0), 191)] = true;
                if(idx < 0) {
                    ++leftPadding;
                }
                if(idx > 191) {
                    ++rightPadding;
                }
            }

            txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange, 1, {leftPadding, rightPadding}), 100e-6);
        }

        TxRxSequence seq(txrxs, {}, 200e-3);
        us4r->setVoltage(30);

        auto[buffer, fcm] = us4r->upload(seq, 2, 14);

        us4r->start();
        for(int i = 0; i < 20; ++i) {
            std::string msg = "i: " + std::to_string(i) + "\n";
            std::cout << msg;
            int16_t* data = buffer->tail(::arrus::devices::HostBuffer::INF_TIMEOUT);
            std::cout << "Got data" << std::endl;
//            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            std::cout << "Processing done." << std::endl;
            buffer->releaseTail(::arrus::devices::HostBuffer::INF_TIMEOUT);
        }
        us4r->stop();

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
