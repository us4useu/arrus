#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>
#include "arrus/core/api/arrus.h"

#include "arrus/core/examples/AfeDemodFIR.h"

enum Commands {
    Write,
    Read,
    Start,
    Exit,
    DemodEn,
    DemodDis,
    DemodDef,
    DemodDec,
    DemodFreq,
    DemodFsweepROI,
    DemodFIR,
    AfeReset,
    Invalid
};

Commands ResolveInput(std::string inp)
{
    if (inp == "wr") return Write;
    if (inp == "rd") return Read;
    if (inp == "exit") return Exit;
    if (inp == "start") return Start;
    if (inp == "afe-rst") return AfeReset;
    if (inp == "demod-en") return DemodEn;
    if (inp == "demod-dis") return DemodDis;
    if (inp == "demod-def") return DemodDef;
    if (inp == "demod-dec") return DemodDec;
    if (inp == "demod-freq") return DemodFreq;
    if (inp == "demod-fsweep") return DemodFsweepROI;
    if (inp == "demod-fir") return DemodFIR;

    return Invalid;
}

int main() noexcept {
    char value = 0;
    std::cin >> value;
    std::cout << "pwi-example " << std::endl;
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {
        // TODO set path to us4r-lite configuration file
        auto settings = ::arrus::io::readSessionSettings("C:/Users/Public/ate_cfg.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();

        ::arrus::BitMask rxAperture(nElements, true);

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{ 0, 8192 };

        std::vector<TxRx> txrxs;

        for (int i = 0; i < 4; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            arrus::BitMask txAperture(nElements, false);
            txrxs.emplace_back(Tx(txAperture, delays, pulse),
                Rx(rxAperture, sampleRange),
                200e-6f);
        }

        TxRxSequence seq(txrxs, {});

        DataBufferSpec outputBuffer{ DataBufferSpec::Type::FIFO, 4 };
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::HOST);

        auto result = session->upload(scheme);

        std::condition_variable cv;
        std::condition_variable cv2;
        using namespace std::chrono_literals;

        std::mutex mutex2;
        std::unique_lock<std::mutex> lock2(mutex2);

        OnNewDataCallback callback = [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
            try {
                std::cout << "Iteration: " << i << ", data: " << std::endl;
                std::cout << "- memory ptr: " << std::hex
                    << ptr->getData().get<short>()
                    << std::dec << std::endl;
                std::cout << "- size: " << ptr->getSize() << std::endl;
                std::cout << "- shape: (" << ptr->getData().getShape()[0] << // calkowita liczba probek
                    ", " << ptr->getData().getShape()[1] << // 32
                    ")" << std::endl;

                std::ofstream file;
                file.open("temp.bin", std::ios::binary | std::ios::out);
                //short* dptr = ptr->getData().getInt16();
                file.write(reinterpret_cast<char *>(ptr->getData().getInt16()/*ptr->getData().get<short>()*/), static_cast<size_t>(ptr->getData().getShape()[0]) * 2);
                file.close();

                //std::string pycmd = "python ./plot.py";
                //system(pycmd.c_str());

                // Stop the system after 10-th frame.
                //ptr->release();

                cv.notify_one();

                cv2.wait(lock2);

                ptr->release();
                ++i;
            }
            catch (const std::exception &e) {
                std::cout << "Exception: " << e.what() << std::endl;
                cv.notify_all();
            }
            catch (...) {
                std::cout << "Unrecognized exception" << std::endl;
                cv.notify_all();
            }
        };

        OnOverflowCallback overflowCallback = [&]() {
            std::cout << "Data overflow occurred!" << std::endl;
            cv.notify_one();
        };

        // Register the callback for new data in the output buffer.
        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        bool cont = true;

        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);


        //print menu
        std::cout << "--- Console options: ---" << std::endl;
        std::cout << " start - start scheme" << std::endl 
            << " wr - write AFE register" << std::endl 
            << " rd - read AFE register" << std::endl 
            << " demod-en - enable AFE demodulator" << std::endl
            << " demod-dis - disable AFE demodulator" << std::endl
            << " demod-def - writes AFE demodulator default config" << std::endl
            << " demod-freq - sets AFE demodulator frequency" << std::endl
            << " demod-fsweep - sets AFE demodulator frequency sweep ROI" << std::endl
            << " demod-fir - writes AFE demodulator decimation filter coefficients" << std::endl
            << " exit - exit app" << std::endl;

        //get number of oems?
        uint8_t nOEMS = 0;
        for (nOEMS = 0; nOEMS < 16; nOEMS++)
        {
            try {
                us4r->getUs4OEM(nOEMS);
            }
            catch (DeviceNotFoundException ex) {

            }
        }

        std::cout << std::endl << "Found " << nOEMS << " OEMs" << std::endl;

        //configure demodulator
        for (uint8_t n = 0; n < nOEMS; n++) {
            us4r->getUs4OEM(n)->enableAfeDemod();
            us4r->getUs4OEM(n)->setAfeDemodDefault();
        }

        session->startScheme();
        cv.wait(lock);

        std::string inp;

        while (cont) {
            //char value = 0;
            std::cout << ">";
            std::cin >> inp;
            switch (ResolveInput(inp)) {
            case Write:
            {
                std::cout << "Register address (hex): " << std::endl;
                std::cin >> inp;
                int regAddr = std::stoi(inp, 0, 16);
                std::cout << "Register value (hex): " << std::endl;
                std::cin >> inp;
                int regVal = std::stoi(inp, 0, 16);
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->setAfe(static_cast<uint8_t>(regAddr), static_cast<uint16_t>(regVal));
                }
                //us4r->getUs4OEM(0)->setAfe(static_cast<uint8_t>(regAddr), static_cast<uint16_t>(regVal));
                break;
            }
            case FIR:
            {
                //writes default 10MHz FIR for now, valid for M = 4
                //us4r->getUs4OEM(0)->writeAfeFIRCoeffs((int16_t*)&fir10M[0], 32);
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->writeAfeFIRCoeffs((int16_t*)&fir10M[0], 32);
                }
                break;
            }
            case DemodEn:
            {
                //us4r->getUs4OEM(0)->enableAfeDemod();
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->enableAfeDemod();
                }
                break;
            }
            case AfeReset:
            {
                //us4r->getUs4OEM(0)->resetAfe();
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->resetAfe();
                }
                break;
            }
            case DemodDis:
            {
                //us4r->getUs4OEM(0)->disableAfeDemod();
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->disableAfeDemod();
                }
                break;
            }
            case DemodDef:
            {
                //us4r->getUs4OEM(0)->setAfeDemodDefault();
                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->setAfeDemodDefault();
                }
                break;
            }
            case DemodDec:
            {
                std::cout << "Decimation factor integer part:" << std::endl;
                std::cin >> inp;
                int integer = std::stoi(inp, 0, 10);

                if (integer < 0 || integer > 0x3F) {
                    std::cout << "Invalid value" << std::endl;
                    break;
                }

                std::cout << "Decimation factor fractional part (0 = 0, 1 = 0.25, 2 = 0.5, 3 = 0.75):" << std::endl;
                std::cin >> inp;
                int quarters = std::stoi(inp, 0, 10);

                if (quarters < 0 || quarters > 3) {
                    std::cout << "Invalid value" << std::endl;
                    break;
                }

                if (quarters == 0) {
                    for (uint8_t n = 0; n < nOEMS; n++) {
                        us4r->getUs4OEM(n)->setAfeDemodDecimationFactor(static_cast<uint16_t>(integer));
                    }
                }
                else {
                    for (uint8_t n = 0; n < nOEMS; n++) {
                        us4r->getUs4OEM(n)->setAfeDemodDecimationFactor(static_cast<uint16_t>(integer), static_cast<uint16_t>(quarters));
                    }
                }
                break;
            }
            case DemodFreq:
            {
                std::cout << "Demodulator frequency in MHz (start freuqency for sweep):" << std::endl;
                std::cin >> inp;
                double startFreq = std::stod(inp);

                std::cout << "Demodulator stop frequency in MHz (enter 0.0 to disable fsweep):" << std::endl;
                std::cin >> inp;
                double stopFreq = std::stod(inp);

                if (stopFreq == 0.0) {
                    for (uint8_t n = 0; n < nOEMS; n++) {
                        us4r->getUs4OEM(n)->setAfeDemodFrequency(startFreq);
                    }
                    //get and output actual set frequency
                    startFreq =  us4r->getUs4OEM(0)->getAfeDemodStartFrequency();
                    std::cout << "Set demodulator frequency = " << starFreq << " MHz" << std::endl;
                }
                else {
                    for (uint8_t n = 0; n < nOEMS; n++) {
                        us4r->getUs4OEM(n)->setAfeDemodFrequency(startFreq, stopFreq);
                    }
                    startFreq = us4r->getUs4OEM(0)->getAfeDemodStartFrequency();
                    std::cout << "Set demodulator start frequency = " << starFreq << " MHz" << std::endl;
                    stopFreq = us4r->getUs4OEM(0)->getAfeDemodStopFrequency();
                    std::cout << "Set demodulator stop frequency = " << stopFreq << " MHz" << std::endl;
                }
                break;
            }
            case DemodFsweepROI:
            {
                std::cout << "Frequency sweep start sample:" << std::endl;
                std::cin >> inp;
                int start = std::stoi(inp, 0, 10);

                if (start < 0 || start > 0xFFFF) {
                    std::cout << "Invalid value" << std::endl;
                    break;
                }

                std::cout << "Frequency sweep stop sample:" << std::endl;
                std::cin >> inp;
                int stop = std::stoi(inp, 0, 10);

                if (stop < 0 || stop > 0xFFFF) {
                    std::cout << "Invalid value" << std::endl;
                    break;
                }

                for (uint8_t n = 0; n < nOEMS; n++) {
                    us4r->getUs4OEM(n)->setAfeDemodFsweepROI(static_cast<uint16_t>(start), static_cast<uint16_t>(stop));
                }
                break;
            }
            case Read:
            {
                std::cout << "Register address (hex): " << std::endl;
                std::cin >> inp;
                int regAddr = std::stoi(inp, 0, 16);
                uint16_t regVal = us4r->getUs4OEM(0)->getAfe(static_cast<uint8_t>(regAddr)); //read only AFE 0 on OEM 0
                std::cout << "Value = " << std::hex << static_cast<int>(regVal) << std::endl;
                break;
            }
            case Start:
                cv2.notify_one();

                break;
            case Exit:
                session->stopScheme();
                cv2.notify_all();
                cont = false;
                break;
            }
        }

        // Wait for callback to signal that we hit 10-th iteration.
        /// tutaj prawdodpobonie proste menu z poziomu konsoli

    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
