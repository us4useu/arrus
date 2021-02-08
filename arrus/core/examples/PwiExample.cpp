#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>

#include <arrus/core/api/arrus.h>

int main() noexcept {
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {
        auto settings = ::arrus::io::readSessionSettings(
                R"(C:\Users\Public\us4r.prototxt)");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        ::arrus::BitMask rxAperture(192, true);
        std::vector<float> delays(192, 0.0f);

        Pulse pulse(4e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};

        std::vector<TxRx> txrxs;

        for(int i = 0; i < 10; ++i) {
            arrus::BitMask txAperture(192, true);
            txrxs.emplace_back(Tx(txAperture, delays, pulse),
                               Rx(txAperture, sampleRange),
                               200e-6f);
        }

        TxRxSequence seq(txrxs, {}, 500e-3f);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::ASYNC);

        auto result = session->upload(scheme);
        us4r->setVoltage(10);

        std::condition_variable cv;
        using namespace std::chrono_literals;

        OnNewDataCallback callback = [&, i = 0](const DataBufferElement::SharedHandle &ptr) mutable {
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
                if(i == 9) {
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
        auto buffer = std::static_pointer_cast<FifoLockFreeBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();

        // Wait for callback to signal that the we hit 10-th iteration.
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
