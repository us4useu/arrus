#include <iostream>
#include <thread>
#include <fstream>
#include <cstdio>
#include <string>
#include <condition_variable>
#include "arrus/core/api/arrus.h"

#include "arrus/core/examples/AfeDemodFIR.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int ac, char *av[]) noexcept {

    char value = 0;
    std::cout << "demod-example. press a key to continue... " << std::endl;
    std::cin >> value;

    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("ddc-dec", po::value<double>(), "Sets DDC decimation factor")
            ("ddc-freq", po::value<double>(), "Sets DDC frequency")
            ("ddc-fir-load", po::value<std::string>()->default_value("../fir.txt")->implicit_value("../fir.txt"), "Loads FIR coefficients from specified file")
            ("ddc-fir-gen", po::value<double>(), "Generates FIR coefficients with specified cutoff frequency")
            ("fname", po::value<std::string>()->default_value("rf.bin")->implicit_value("rf.bin"), "Filename for output data")
            ;

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 0;
        }

        uint32_t sampleOffset;
        double decFactor;
        double ddcFreq;
        double ddcFirCutoff = -255.0;
        std::string fname, firFname;

        if (vm.count("ddc-dec"))
        {
            decFactor =  vm["ddc-dec"].as<double>();

            if (decFactor < 2.0 || decFactor > 63.75) {
                std::cout << "Invalid decimation factor (too low/high)" << std::endl;
                return 0;
            }
            if ((decFactor - (long)decFactor) == 0.25) {}
            else if ((decFactor - (long)decFactor) == 0.5) {}
            else if ((decFactor - (long)decFactor) == 0.75) {}
            else if ((decFactor - (long)decFactor) == 0.0) {}
            else
            {
                std::cout << "Invalid decimation factor fraction - " << (decFactor - (long)decFactor) <<  std::endl;
                return 0;
            }

            sampleOffset = 34 + (uint32_t)(16 * decFactor);
        }
        if (vm.count("ddc-freq"))
        {
            ddcFreq = vm["ddc-freq"].as<double>();
        }
        if (vm.count("ddc-fir-gen"))
        {
            ddcFirCutoff = vm["ddc-fir-gen"].as<double>();
            firFname = "fir_generated.txt";
        }
        else if (vm.count("ddc-fir-load"))
        {
            firFname = vm["ddc-fir-load"].as<std::string>();
        }
        if (vm.count("fname"))
        {
            fname = vm["fname"].as<std::string>();
        }


            // TODO set path to us4r-lite configuration file
        auto settings = ::arrus::io::readSessionSettings("C:/Users/user/us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();

        ::arrus::BitMask rxAperture(nElements, true);

        Pulse pulse(8.125e6, 2, false);
        // with the hardware decimation turned on proper rx data starts at sample 98.
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{ sampleOffset, (8192 + sampleOffset) };

        std::vector<TxRx> txrxs;

        for (int i = 0; i < 1; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            arrus::BitMask txAperture(nElements, true);
            txrxs.emplace_back(Tx(txAperture, delays, pulse),
                Rx(rxAperture, sampleRange),
                1000e-6f);
        }

        TxRxSequence seq(txrxs, {});

        DataBufferSpec outputBuffer{ DataBufferSpec::Type::FIFO, 4 };
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::HOST);

        auto result = session->upload(scheme);
        us4r->setVoltage(10);
        //run fir generation script if requested
        if (ddcFirCutoff > 0)
        {
            std::string pycmd = "python ../AfeDDCExample_v2_FirGen.py " + std::to_string(decFactor) + " " + std::to_string(ddcFirCutoff);
            system(pycmd.c_str());
        }

        //read fir from file
        int16_t fCoeffs[1024]; //arbitrary buffer size
        uint16_t numCoeffs = 0;

        std::fstream firfile(firFname, std::ios_base::in);
        while (firfile >> fCoeffs[numCoeffs])
        {
            numCoeffs++;
        }
        
        //configure demodulator
        us4r->setAfeDemod(ddcFreq, decFactor, fCoeffs, numCoeffs);

        std::condition_variable cv;
        using namespace std::chrono_literals;

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

                //dump data to file
                std::ofstream file;
                file.open(fname, std::ios::binary | std::ios::out);
                file.write(reinterpret_cast<char *>(ptr->getData().getInt16()/*ptr->getData().get<short>()*/), static_cast<size_t>(ptr->getSize()));
                file.close();

                // Stop the system after receiving frame.
                cv.notify_one();
                ptr->release();
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

        std::cout << "press a key to startScheme... " << std::endl;
        std::cin >> value;

        session->startScheme();

        // Wait for callback to signal that we got frame.
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);

        session->stopScheme();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
