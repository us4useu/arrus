#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>

#include "arrus/core/api/arrus.h"

int main() noexcept {
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {
        auto *logging = ::arrus::useDefaultLoggerFactory();
        logging->setClogLevel(::arrus::LogSeverity::TRACE);
        std::shared_ptr<std::ofstream> logFile = std::make_shared<std::ofstream>("log.log");
        logging->addOutputStream(logFile, ::arrus::LogSeverity::TRACE);
        auto settings = ::arrus::io::readSessionSettings("C:/Users/Public/us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
	us4r->disableHV();
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask rxAperture(nElements, true);

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};

        std::vector<TxRx> txrxs;

		// 10 plane waves
        for(int i = 0; i < 1; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            for(int d = 0; d < nElements; ++d) {
                delays[d] = d*1e-9f;
            }
            arrus::BitMask txAperture(nElements, false); // No TX.
            txrxs.emplace_back(Tx(txAperture, delays, pulse), Rx(rxAperture, sampleRange), 200e-6f);
        }

        TxRxSequence seq(txrxs, {}, 100e-3);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 16};
        Scheme scheme(seq, 16, outputBuffer, Scheme::WorkMode::ASYNC);
        std::cout << "SYNC mode" << std::endl;

        auto result = session->upload(scheme);
		us4r->disableHV();

        std::condition_variable cv;
        using namespace std::chrono_literals;

        OnNewDataCallback callback = [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
            try {
                std::cout << "Iteration: " << i << ", data: " << std::endl;
                std::cout << "- memory ptr: " << std::hex
                                           << ptr->getData().get<short>()
                                           << std::dec << std::endl;
                std::cout << "- size: " << ptr->getSize() << std::endl;
                std::cout << "- shape: (" << ptr->getData().getShape()[0] <<
                                     ", " << ptr->getData().getShape()[1] <<
                                     ")" << std::endl;

                // Stop the system after 10-th frame.
                if(i == 100) {
                    cv.notify_one();
                }
                std::this_thread::sleep_for(150ms);
                ptr->release();
                ++i;
            } catch(const std::exception &e) {
                std::cout << "Exception: " << e.what() << std::endl;
                cv.notify_all();
            } catch (...) {
                std::cout << "Unrecognized exception" << std::endl;
                cv.notify_all();
            }
        };

        OnOverflowCallback overflowCallback = [&] () {
            std::cout << "Data overflow occurred!" << std::endl;
            cv.notify_one();
        };

        // Register the callback for new data in the output buffer.
        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();

        // Wait for callback to signal that we hit 10-th iteration.
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);

        // Stop the system.
        session->stopScheme();

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
