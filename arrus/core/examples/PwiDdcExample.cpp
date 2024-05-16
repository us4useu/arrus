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
        arrus::useDefaultLoggerFactory()->setClogLevel(arrus::LogSeverity::INFO);
        auto settings = ::arrus::io::readSessionSettings("us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R*) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask aperture(nElements, true);
        float txFrequency = 6e6; // [MHz]
        Pulse pulse(txFrequency, 2, false);
        // NOTE: we specify here the NUMBER OF I/Q SAMPLES to acquire 
        // (i.e. the target sampling frequency = fs/decimation factor is assumed).
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 512}; 

        std::vector<TxRx> txrxs;
        // 10 plane waves
        for(int i = 0; i < 10; ++i) {
            std::vector<float> delays(nElements, 0.0f);
            txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange), 200e-6f);
        }

        TxRxSequence seq(txrxs, {}, 100e-3, 1);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
        // Digital Down Conversion:
        // FIR filter coefficients:
        // - the FIR filter order should be equal decimationFactor*16
        // - only a upper half of the FIR filter coefficients should be provided.
        // 
        float decimationFactor = 4.0f;
        // NOTE: the below FIR coefficients were calculated using Python: 
        // filter_order = decimationFactor*16
        // coeffs = scipy.signal.firwin(filter_order, txFrequency, fs=65e6)
        // coeffs = coeffs[filter_order//2:]
        std::vector<float> firCoefficients = {0.18234651659672152, 0.16172486625099816, 0.12487982587460944, 0.07944398046616387, 0.03430398844523893, -0.0026133185074908405, -0.026157255063119715, -0.034889180817011325, -0.030945327136370222, -0.018965928416555058, -0.004494915529298055, 0.007628287152588109, 0.014419713683593693, 0.015175743942293598, 0.011161684805841312, 0.00478318006743135, -0.001412573813589476, -0.005562384563359233, -0.006912138093338076, -0.005787361358840273, -0.0032172403273668768, -0.0004159330245921233, 0.0016683945062905931, 0.0025961471738894463, 0.0024366998597999934, 0.0015898600953486795, 0.0005435013516173024, -0.0003223898280114102, -0.0008232837583015619, -0.0009500466921633298, -0.000789093632050986, -0.00044401971096737745};
        float demodulationFrequency = txFrequency;
        DigitalDownConversion ddc(demodulationFrequency, firCoefficients, decimationFactor);

        Scheme scheme(seq, 4, outputBuffer, Scheme::WorkMode::HOST, ddc);

        auto result = session->upload(scheme);
        us4r->setDtgcAttenuation(24);
        us4r->setVoltage(5);

        std::condition_variable cv;
        using namespace std::chrono_literals;

        OnNewDataCallback callback = [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
            try {
                std::cout << "Iteration: " << i << ", data: " << std::endl;
                std::cout << "- memory ptr: " << std::hex
                          << ptr->getData().get<short>()
                          << std::dec << std::endl;
                std::cout << "- size: " << ptr->getSize() << std::endl;
                std::cout << "- shape (total number of samples, I/Q, RX channels): (" 
                          << ptr->getData().getShape()[0] 
                          << ", " << ptr->getData().getShape()[1] 
                          << ", " << ptr->getData().getShape()[2] 
                          << ")" << std::endl;
                // Stop the system after 10-th frame.
                if(i == 10) {
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
