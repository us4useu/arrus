#include <chrono>
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
        auto settings = ::arrus::io::readSessionSettings("/path/to/us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto ultrasound = (::arrus::devices::Us4R*) session->getDevice("/Ultrasound:0");
        auto probe = ultrasound->getProbe(0);

        // Setting voltages for levels: -1: 5, +1: 6, -2: 10, +2: 11.
	    ultrasound->setVoltage({HVVoltage(5, 6), HVVoltage(10, 11)});


        unsigned nElements = probe->getModel().getNumberOfElements().product();
        ::arrus::BitMask aperture(nElements, true);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};

        WaveformBuilder waveformBuilder{};

        // Set states -1 (0.2 us), 1 (0.5 us), -1 (1 us), and repeat that twice.
        waveformBuilder.add(
            WaveformSegment {
                {0.2e-6f, 0.5e-6f, 1e-6f}, // durations
                {-1,      1,       -1}     // states (levels)
            },
            2 // how many times to repeat this segment
        );

        // Set states 2 (1.5 us), 0 (2 us), 2 (3 us), run it only once (n repeats = 1).
        waveformBuilder.add(
            WaveformSegment {
                {1.5e-6f, 2e-6f, 3e-6f}, // durations
                {2,      0,   2}         // states (levels)
            }
        );

        auto waveform = waveformBuilder.build();

        std::vector<TxRx> txrxs;

        for(int i = 0; i < 2; ++i) {
            std::vector<float> delays(nElements, 0.0f);

            // Set the waveform in the given TXs.
            txrxs.emplace_back(Tx(aperture, delays, waveform), Rx(aperture, sampleRange), 200e-6f);
        }

        TxRxSequence seq(txrxs, {});
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
                if(i == 30) {
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

        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();

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
