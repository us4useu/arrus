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
        arrus::useDefaultLoggerFactory()->setClogLevel(arrus::LogSeverity::TRACE);
        auto settings = ::arrus::io::readSessionSettings("/home/pjarosik/tmp/test.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto ultrasound = (::arrus::devices::Ultrasound*) session->getDevice("/Ultrasound:0");
        auto probe = ultrasound->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask aperture(nElements, false);
        for(int i = 0; i < 64; ++i) {
            aperture[i] = true;
        }

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 1024};

        std::vector<TxRx> txrxs;

        // 10 plane waves
        for(int i = 0; i < 175; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange), 200e-6f);
        }

        TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI, 1);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
        Scheme scheme(seq, 4, outputBuffer, Scheme::WorkMode::HOST);

        auto result = session->upload(scheme);

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
                if(i == 3) {
                    cv.notify_one();
                }
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