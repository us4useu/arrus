#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>
#include "arrus/core/api/framework/FifoLockfreeBuffer.h"

#include "arrus/core/api/io/settings.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/common/logging.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"
#include "arrus/core/api/framework/DataBufferSpec.h"

int main() noexcept {
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
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

        for(int i = 0; i < 10; ++i) {
            arrus::BitMask aperture(192, true);
            txrxs.emplace_back(Tx(aperture, delays, pulse),
                               Rx(aperture, sampleRange, 1, {0, 0}),
                               100e-6f);
        }

        TxRxSequence seq(txrxs, {}, 50e-3f);
        DataBufferSpec outputBuffer{DataBufferType::FIFO_LOCKFREE, 10};
        Scheme scheme(seq, 2, outputBuffer);

        auto result = session->upload(scheme);
        us4r->setVoltage(10);

        std::condition_variable cv;
        using namespace std::chrono_literals;

        OnNewDataCallback callback = [&, i = 0](const DataBufferElement::SharedHandle &ptr) mutable {
            try {
                std::cout << "Callback!" << std::endl;
                if(i == 9) {
                    cv.notify_one();
                }
                ptr->release();
                std::cout << ptr->getData().getShape().size() << std::endl;
                std::cout << ptr->getData().getShape().get(0) << ", " << ptr->getData().getShape().get(1)<< std::endl;
                ++i;
            } catch(const std::exception &e) {
                std::cout << "Exception: " << e.what() << std::endl;
                cv.notify_one();
            }
        };

        OnOverflowCallback overflowCallback = [&] () {
            std::cout << "Overflow!!!" << std::endl;
            cv.notify_one();
        };

        // Register the callback for new data in the output buffer.
        auto buffer = std::static_pointer_cast<FifoLockfreeBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);
        std::cout << "Closing scheme and program." << std::endl;
        session->stopScheme();
    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
