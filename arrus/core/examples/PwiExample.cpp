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
    FIR,
    AfeReset,
	Invalid
};

Commands ResolveInput(std::string inp)
{
	if (inp == "wr") return Write;
	if (inp == "rd") return Read;
	if (inp == "exit") return Exit;
	if (inp == "start") return Start;
    if (inp == "fir") return FIR;
    if (inp == "afe-rst") return AfeReset;
    if (inp == "demod-en") return DemodEn;
    if (inp == "demod-dis") return DemodDis;
    if (inp == "demod-def") return DemodDef;

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
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};

        std::vector<TxRx> txrxs;

        for(int i = 0; i < 4; ++i) {
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
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
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
                file.write(reinterpret_cast<char *>(ptr->getData().getInt16()/*ptr->getData().get<short>()*/), static_cast<size_t>(ptr->getData().getShape()[0])*2);
                file.close();

                std::string pycmd = "python ./plot.py";
                system(pycmd.c_str());

                // Stop the system after 10-th frame.
                //ptr->release();
               
                cv.notify_one();

                cv2.wait(lock2);
                
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

        bool cont = true;

        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);


        //print menu
        std::cout << "--- Console options: ---" << std::endl;
        std::cout << " wr - write AFE register" << std::endl << " rd - read AFE register"
            << std::endl << " start - start scheme" << std::endl << " exit - exit app" << std::endl;


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
                    int regAddr = stoi(inp, 0, 16);
                    std::cout << "Register value (hex): " << std::endl;
                    std::cin >> inp;
                    int regVal = stoi(inp, 0, 16);
                    us4r->getUs4OEM(0)->setAfe(static_cast<uint8_t>(regAddr), static_cast<uint16_t>(regVal));
					break;
				}
                case FIR:
                {
                    //writes default 10MHz FIR for now, valid for M = 4
                    us4r->getUs4OEM(0)->writeAfeFIRCoeffs((int16_t*)&fir10M[0], 32);
                    break;
                }
                case DemodEn:
                {
                    us4r->getUs4OEM(0)->enableAfeDemod();
                    break;
                }
                case AfeReset:
                {
                    us4r->getUs4OEM(0)->resetAfe();
                    break;
                }
                case DemodDis:
                {
                    us4r->getUs4OEM(0)->disableAfeDemod();
                    break;
                }
                case DemodDef:
                {
                    us4r->getUs4OEM(0)->setAfeDemodDefault();
                    break;
                }
				case Read: 
				{	
					std::cout << "Register address (hex): " << std::endl;
					std::cin >> inp;
					int regAddr = stoi(inp, 0, 16);
					uint16_t regVal = us4r->getUs4OEM(0)->getAfe(static_cast<uint8_t>(regAddr));
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

    } catch(const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
