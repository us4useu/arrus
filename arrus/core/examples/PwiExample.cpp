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
    using namespace ::arrus::framework::pipeline;
    try {
        // TODO set path to us4r-lite configuration file
        auto settings = ::arrus::io::readSessionSettings("C:/Users/Public/us4r.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");
        auto probe = us4r->getProbe(0);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask rxAperture(nElements, true);

        Pulse pulse(6e6, 2, false);
        ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 8192};

        std::vector<TxRx> txrxs;

        // 10 plane waves
        for (int i = 0; i < 4; ++i) {
            // NOTE: the below vector should have size == probe number of elements.
            // This probably will be modified in the future
            // (delays only for active tx elements will be needed).
            std::vector<float> delays(nElements, 0.0f);
            for (int d = 0; d < nElements; ++d) {
                delays[d] = d * i * 1e-9f;
            }
            arrus::BitMask txAperture(nElements, true);
            txrxs.emplace_back(Tx(txAperture, delays, pulse), Rx(txAperture, sampleRange), 200e-6f);
        }

        //        TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI, 32);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};

        // TODO work mode?
        // moze zamiast work mode, session powinna udostepniac metody, ktore pozwalaja wykonac sekwencje raz?
        // np. sesion.run -> wszystkie wyjscia (jakie?)
        // np. session.eval(operacja) ->
        // mozna w kazdym podgrafie ustawic trigger source
        // - w zrodlach mozna ustawic trigger source
        // metody sesji:
        // session.start() -> cos musi wystartowac? co? jakis generator triggera?
        // - np. session.start(triggerSource), i to bedzie powodowac
        // - np. session.eval(output), i to bedzie powodowac wykonanie wszystkiego, co prowadzi do danego outputu
        // - czyli Pipeline powininen udostepniac output (wskazac krawedz wyjsciowa, ktora nalezy zewaluowac)

        // wewnetrzny bufor? -> ustawienie w us4R, ile elementow ma byc w buforze
        // dodatkowy setter

        // Zaimplementowany graf powinno dac sie zseralizowac (np. do hdf5 zeby zwrocic w metadanych)

        // Problem z okresleniem trigger source to
        // poza tym, ten czasami moze przydac sie rozne czasy pomiedzy triggerami
        // wiec to chyba powinno zostac parametrem TxRx (jako czas, ile ma zajac dany TX/RX)

        // utworz/odczytaj opis graphu (tylko deskrypcyjnie)
        // zaladuj graph (skonfiguruj urzadzenia do wykonania grafu: utworz bufory, skonfiguruj )
        //
        // Utworz graf przetwarzania
        // zuploaduj graf przetwarzania
        // Graph graph;
        // Pipeline {

        // }
        // session.upload(graph);
        // wykonaj jakis kawalek grafu
        // session

        Pipeline pipeline{
            Pipeline {
                {
                    TxRx{},
                    TxRx{},
                    TxRx{},
                    Repeat(10),
                    DigitalDownConversion(),
                    Enqueue{outputBuffer},
                    // timinigi: PRI, SRI, BRI: gdzie powinny byc specyfikowane?
                    // - kto wywoluje "eval" ?
                    // -- uzytkownik powinien miec taka mozliwosc
                    // -- powinna byc rowniez mozliwosc uruchomienia zewnetrznego trigger generatora (dzialajacego na zewnetrznym urzadzeniu) - to bedzie generator z us4OEM
                    // ---- Inaczej mowiac, trigger generator to tez operacja, ktora ma swoj placement
                    // ---- np.
                    // rozmiar batcha: jak wyspecyfikowac, gdybym uzywal tutaj pojedynczych tx/rx? (Step: Loop?)
                    // - step "Repeat", ktory bedzie pozwalal na wykonanie jakiegos kawalka wiele razy?
                    // - na jaki operator w grafie to sie przelozy?
                    // - operator Loop(nTimes, goto: "nazwa_operatora")
                    // --
                    // enqueue/dequeue: czy podawac tutaj enque/deque?
                    // - tak, w przypadku Pipeline ladowanego na Us4R, sekwencja operacji musi sie zakonczyc Enqueue
                    // - pozwoli to np. na reczne wywolywanie operatorow z Pipeline GPU, bez potrzeby tworzenia buforow
                },
                "Ultrasound:0"
            },
            Pipeline {
                {
                    Dequeue{outputBuffer},

                    RxBeamforming{}
                },
                "GPU:0"
            }
        };

//        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::HOST);
        session->start(Pipe)

        auto result = session->upload(scheme);
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
