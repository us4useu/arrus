#include <arrus/core/api/arrus.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>

constexpr auto NUM_CHANNELS = 64u;

arrus::devices::ProbeModel createProbe() {
    constexpr double curvature_radius =
        std::numeric_limits<double>::infinity(); // flat linear array

    const double probe_pitch{0.205f};
    const auto num_elements =
        static_cast<arrus::devices::ProbeModel::ElementIdxType>(NUM_CHANNELS);

    const arrus::devices::ProbeModelId probe_id{"us4us", "dummy"};
    const arrus::Interval<float> tx_frequency_range{3e6f, 16e6f}; // NOLINT
    const arrus::Interval<arrus::Voltage> voltage_range{5, 50};   // NOLINT

    return arrus::devices::ProbeModel{probe_id,
                                      arrus::Tuple{num_elements},
                                      arrus::Tuple{probe_pitch},
                                      tx_frequency_range,
                                      voltage_range,
                                      curvature_radius};
}

arrus::devices::ProbeAdapterSettings::ChannelMapping
createDummyChannelMapping() {
    using ProbeAdapterSettings = arrus::devices::ProbeAdapterSettings;

    ProbeAdapterSettings::ChannelMapping channel_mapping;
    channel_mapping.reserve(NUM_CHANNELS);
    std::generate_n(
        std::back_inserter(channel_mapping), NUM_CHANNELS, [n = 0u]() mutable {
            ProbeAdapterSettings::ChannelAddress address{
                static_cast<arrus::devices::Ordinal>(n / (NUM_CHANNELS / 2)),
                static_cast<arrus::ChannelIdx>(n)};
            ++n;
            return address;
        });
    return channel_mapping;
}

arrus::session::SessionSettings createSettings() {
    const arrus::devices::ProbeAdapterModelId probe_adapter_model{"us4us",
                                                                  "dummy"};
    const auto channel_mapping = createDummyChannelMapping();

    const arrus::devices::ProbeAdapterSettings probe_adapter_settings{
        probe_adapter_model,
        static_cast<arrus::ChannelIdx>(channel_mapping.size()), channel_mapping};

    const auto transducer = createProbe();
    std::vector<arrus::ChannelIdx> probe2adapter_mapping(
        transducer.getNumberOfElements().get(0));
    std::iota(probe2adapter_mapping.begin(), probe2adapter_mapping.end(),
              arrus::ChannelIdx{0});
    const arrus::devices::ProbeSettings probe_settings{transducer,
                                                       probe2adapter_mapping};

    // the settings below do not matter as we are not interested in the actual
    // acquired data they just need to be valid values
    arrus::devices::RxSettings rx_settings{
        std::nullopt, static_cast<uint16_t>(24),   static_cast<uint16_t>(24),
        {},           static_cast<uint32_t>(15e6), static_cast<uint16_t>(50)};

    const arrus::devices::HVModelId high_voltage_model{"us4us", "hv256"};

    const arrus::devices::Us4RSettings settings{
        probe_adapter_settings,
        probe_settings,
        rx_settings,
        arrus::devices::HVSettings{high_voltage_model},
        {},        // no probe channels mask
        {{}, {}}}; // no us4OEM channel masking

    return arrus::session::SessionSettings{settings};
}

arrus::ops::us4r::Scheme createScheme() {
    using Scheme = arrus::ops::us4r::Scheme;

    const arrus::ops::us4r::Pulse pulse{6.25e6, 3, false};
    const std::pair<arrus::uint32, arrus::uint32> sample_range{0, 2048};
    constexpr float pulse_repetition_fequency = 6000.0f;
    constexpr float pulse_repetition_interval = 1.0f/ pulse_repetition_fequency;
    constexpr int num_plane_waves = 128;
    const std::vector<float> delays(NUM_CHANNELS, 0.0f);
    const arrus::BitMask tx_aperture(NUM_CHANNELS, true);

    std::vector<arrus::ops::us4r::TxRx> tx_rx_events;
    tx_rx_events.reserve((num_plane_waves));
    std::generate_n(std::back_inserter(tx_rx_events), num_plane_waves, [&]() {
        return arrus::ops::us4r::TxRx{
            arrus::ops::us4r::Tx(tx_aperture, delays, pulse),
            arrus::ops::us4r::Rx(tx_aperture, sample_range),
            pulse_repetition_interval};
    });

    arrus::ops::us4r::TxRxSequence us4us_sequence{std::move(tx_rx_events), {}};
    arrus::framework::DataBufferSpec output_buffer{
        arrus::framework::DataBufferSpec::Type::FIFO, 128};
    return Scheme{us4us_sequence, 128, output_buffer, Scheme::WorkMode::ASYNC};
}

int main() noexcept {
    using namespace arrus::session;
    using namespace arrus::devices;
    using namespace arrus::ops::us4r;
    using namespace arrus::framework;
    try {
        // TODO set path to us4r-lite configuration file
        auto settings = createSettings();
        auto session = arrus::session::createSession(settings);
        if (auto us4r = dynamic_cast<Us4R *>(session->getDevice("/Us4R:0"));
            us4r != nullptr) {
            us4r->setVoltage(5);
        }

        std::cout << "Probe with " << NUM_CHANNELS << " elements." << std::endl;

        auto scheme = createScheme();

        auto result = session->upload(scheme);
       
        std::condition_variable cv;
        using namespace std::chrono_literals;

        OnNewDataCallback callback =
            [&, i = 0](const BufferElement::SharedHandle &ptr) mutable {
                try {
                    std::cout << "    Iteration: " << i << std::endl;
                    if (i == 10) {
                        cv.notify_one();
                    }
//                    std::this_thread::sleep_for(1s);
                    ptr->release();
                    ++i;
                } catch (const std::exception &e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                    cv.notify_all();
                } catch (...) {
                    std::cout << "Unrecognized exception" << std::endl;
                    cv.notify_all();
                }
            };

        OnOverflowCallback overflowCallback = [&]() {
            std::cout << "Data overflow occurred!" << std::endl;
            cv.notify_one();
        };

        // Register the callback for new data in the output buffer.
        auto buffer = std::dynamic_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        if (auto us4r = dynamic_cast<Us4R *>(session->getDevice("/Us4R:0"));
            us4r != nullptr) {
            us4r->setStandardIODriveMode();
            us4r->setIOLevels(0b00000000);
            //enable waveform drive mode
            us4r->setWaveformIODriveMode();
			us4r->setIOBSRegister(0, 0, 0, false, 4);
			us4r->setIOBSRegister(0, 1, 0b00000010, false, 4);
			us4r->setIOBSRegister(0, 2, 0b00000011, false, 4);
			us4r->setIOBSRegister(0, 3, 0, false, 4);
			us4r->setIOBSRegister(0, 4, 0b00001000, false, 4);
			us4r->setIOBSRegister(0, 5, 0, true, 4);
            //write IO waveform 0 (ramp)
            for (uint8_t firing = 0; firing < 16; firing++) {
                us4r->setFiringIOBS(firing, 0);
            }
        }

        std::cout << "- Starting " << std::endl;
        session->startScheme();

        // Wait for callback to signal that we hit 10-th iteration.
        std::mutex mutex;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock);

        // Stop the system.
        std::cout << "STOPPING SCHEME" << std::endl;
        session->stopScheme();
        std::cout << "SCHEME STOPPED" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));

        if (auto us4r = dynamic_cast<Us4R *>(session->getDevice("/Us4R:0"));
            us4r != nullptr) {
            std::cout << "conf = " << std::hex << us4r->getSequencerConfRegister() << std::endl;
            std::cout << "ctrl = " << std::hex << us4r->getSequencerCtrlRegister() << std::endl;
        }

       /* std::cout << "us4OEM periph list:" << std::endl;
        if (auto us4r = dynamic_cast<Us4R *>(session->getDevice("/Us4R:0"));
            us4r != nullptr) {
            us4r->listPeriphs();
        }

        std::cout << "Type periph ID to dump to file" << std::endl;
        int periphId;
        std::cin >> periphId;
        if (auto us4r = dynamic_cast<Us4R *>(session->getDevice("/Us4R:0"));
            us4r != nullptr) {
            us4r->dumpPeriph("periphDump", periphId);
        }*/

    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    std::cout << "Press enter key to exit" << std::endl;
    std::string value;
    std::cin >> value;

    return 0;
}