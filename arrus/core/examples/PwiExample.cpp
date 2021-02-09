#include <iostream>
#include <thread>
#include <condition_variable>

#include <arrus/core/api/arrus.h>

int main() noexcept {
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {
    	// Read session configuration from the file.
        auto settings = ::arrus::io::readSessionSettings(
                R"(C:\Users\Public\us4r.prototxt)");
        // Create new session.
        auto session = ::arrus::session::createSession(settings);

        // Get Us4R device handle.
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

        // Tx/Rx sequence:
        // Common Tx parameters:
        ::arrus::BitMask rxAperture(192, true);
		Pulse pulse(4e6, 2, false);

		// Common Rx parameters:
		std::vector<float> delays(192, 0.0f);
		arrus::BitMask txAperture(192, true);
		float pri = 200e-6f;
		::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};
        std::vector<TxRx> txrxs;
        for(int i = 0; i < 10; ++i) {
            txrxs.emplace_back(Tx(txAperture, delays, pulse),
                               Rx(txAperture, sampleRange),
                               200e-6f);
        }
        TxRxSequence seq(txrxs, {}, 500e-3f);

        // Define RF channel data output buffer.
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
        // Define scheme to execute.
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::ASYNC);

        // Upload the scheme.
        auto result = session->upload(scheme);
        // Set HV voltage.
        us4r->setVoltage(10);

        // Create "on new data" callback function.
        // In this example, the callback function counts the number of frames
        // that currently occurred and stops the session when a 10th frame is
        // acquired.
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

        // Create callback to be called when overflow occurs.
        OnOverflowCallback overflowCallback = [&] () {
            std::cout << "Data overflow occurred!" << std::endl;
            cv.notify_one();
        };

        // Register callbacks in the data buffer.
        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        // Start the scheme.
        session->startScheme();
        // At this point, data acquisition is started
        // (the occurrence of new data is signaled by the callback function).

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
