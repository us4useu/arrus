// Smallest ARRUS acquisition: 1 TX/RX event, stops after the first frame.
// Derived from arrus/core/examples/PwiExample.cpp, reduced for a bring-up bench.
//
// SCOPE: this is a BENCH DIAGNOSTIC for the Ethernet/Holoscan port, not a general example.
// It assumes a session with no HV and no physical probe, and the frame-inspection block below
// is tied to the CURRENT, UNFINISHED state of the Ethernet data path (see FRAME LAYOUT).
// Once us4r-api strips the header and normalises byte order, that block must be revisited.
//
// Differences from PwiExample that matter on a bench with no HV and no probe:
//   - no setVoltage() call (throws "No HV have been set." with no hv: block)
//   - one TX/RX instead of two
//   - the RX test pattern is switched on explicitly (see setTestPattern below)
//
// SAMPLE RANGE {1, 1025} IS DELIBERATE. ScheduleReceive's `start` param is
//     sampleRxOffset + startSampleRaw
//   = descriptor.getSampleTxStart() + startSample * rxDecimationFactor       (Us4OEMImpl.cpp:327-337)
//   = 35 + 1*1 = 36 on OEM+
// 36 is the ACQ_RX_DELAY measured as correct on this bench; the default {0, 1024} yields 35,
// which was measured to leave one corrupt sample row at the head of the capture.
//
// The sample COUNT is the difference, not the endpoint:
//     nSamples = endSample - startSample = 1024        (Us4OEMImpl.cpp:329)
// so widening the range to {1, 1025} keeps 1024 samples / 65536 bytes per firing.
//
// SetRxDelay (the separate TIME register, fed by Us4RImpl::getRxDelay) is a different
// axis and does not bear on this - left as ARRUS computes it.

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <utility>
#include <vector>

#include "arrus/core/api/arrus.h"

int main() noexcept {
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;
    try {
        arrus::useDefaultLoggerFactory()->setClogLevel(arrus::LogSeverity::TRACE);

        auto settings = ::arrus::io::readSessionSettings("./us4r_eth_bench.prototxt");
        auto session = ::arrus::session::createSession(settings);
        auto ultrasound = (::arrus::devices::Us4R *) session->getDevice("/Ultrasound:0");
        auto probe = ultrasound->getProbe(0);

        // NOTE: no setVoltage() - there is no HV in this configuration.

        // ARRUS calls DisableTestPatterns during init, so without this the AFE returns real
        // (unsignalled) noise - which cannot distinguish a correct transfer from a corrupt one.
        // RAMP makes the payload verifiable end to end.
        ultrasound->setTestPattern(::arrus::devices::Us4OEM::RxTestPattern::RAMP);

        unsigned nElements = probe->getModel().getNumberOfElements().product();
        std::cout << "Probe with " << nElements << " elements." << std::endl;

        ::arrus::BitMask aperture(nElements, true);
        std::vector<float> delays(nElements, 0.0f);

        Pulse pulse(6e6, 2, false);
        std::pair<::arrus::uint32, ::arrus::uint32> sampleRange{1, 1025};

        // PRI kept generous: the floor is 35 us (TxRxLimits), but ECB round trips
        // are ~0.5 ms, so there is no reason to run tight during bring-up.
        std::vector<TxRx> txrxs;
        txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange), 1000e-6f);

        TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI, 1);
        DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 2};
        Scheme scheme(seq, 2, outputBuffer, Scheme::WorkMode::HOST);

        auto result = session->upload(scheme);

        std::mutex mutex;
        std::condition_variable cv;
        // Tracked separately: an overflow also ends the wait, but it is NOT a successful capture
        // and must not be reported as one.
        bool frameReceived = false;
        bool overflowed = false;

        OnNewDataCallback callback = [&](const BufferElement::SharedHandle &ptr) {
            try {
                auto &data = ptr->getData();
                // Derive the dimensions from the buffer rather than assuming them, so that
                // changing sampleRange above cannot silently desynchronise these loops.
                const size_t nRows = (size_t) data.getShape()[0];
                const size_t nChannels = (size_t) data.getShape()[1];
                std::cout << "Frame received."
                          << " size=" << ptr->getSize()
                          << " shape=(" << nRows << ", " << nChannels << ")" << std::endl;
                std::cout << "First 16 samples of channel 0: ";
                for (size_t j = 0; j < 16 && j < nRows; ++j) {
                    std::cout << data.get<short>(j, 0) << " ";
                }
                std::cout << std::endl;
                // FRAME LAYOUT: the first 64 bytes are the us4OEM Data_Receiver DIAGNOSTIC header
                // (Data_Receiver_IP.sv emits debug_sample for sample_counter 0 and 1), written into
                // the DDR4 capture by the RTL. It is therefore part of the us4OEM frame format on
                // EVERY transport, not an Ethernet artifact. It occupies exactly one row
                // (64 B = 32ch x int16), which is why a 1024-row capture yields 1023 sample rows.
                // This raw consumer does not skip it; a real pipeline handles frame metadata itself.
                //
                // BYTE ORDER: on this Ethernet bench the samples arrive big-endian. Whether that is
                // a genuine eth-vs-PCIe difference (and so us4r-api's to correct) was open at the
                // time of writing. Un-swapping here is DIAGNOSTIC, to prove the payload is an intact
                // ramp - it is not an assertion about the contract.
                auto bswap = [](short v) -> short {
                    unsigned short u = (unsigned short) v;
                    return (short) (unsigned short) ((u << 8) | (u >> 8));
                };
                std::cout << "HEADER row 0 (raw int16 x" << nChannels << "):";
                for (size_t ch = 0; ch < nChannels; ++ch) std::cout << " " << data.get<short>(0, ch);
                std::cout << std::endl;

                // Per-channel: does the byte-swapped stream form a clean +1 ramp over rows 1..N-1?
                size_t chClean = 0; long badTotal = 0;
                bool haveBad = false;
                size_t firstBadCh = 0, firstBadRow = 0; int firstBadGot = 0, firstBadWant = 0;
                for (size_t ch = 0; ch < nChannels; ++ch) {
                    long badHere = 0;
                    for (size_t j = 2; j < nRows; ++j) {
                        int prev = (int)(unsigned short) bswap(data.get<short>(j - 1, ch));
                        int cur  = (int)(unsigned short) bswap(data.get<short>(j, ch));
                        int want = (prev + 1) & 0xFFFF;
                        if (cur != want) {
                            ++badHere;
                            if (!haveBad) {
                                haveBad = true;
                                firstBadCh = ch; firstBadRow = j; firstBadGot = cur; firstBadWant = want;
                            }
                        }
                    }
                    if (badHere == 0) ++chClean; else badTotal += badHere;
                }
                std::cout << "UNSWAPPED: channels forming a clean +1 ramp: " << chClean << "/"
                          << nChannels << "   total deviations: " << badTotal << std::endl;
                if (haveBad)
                    std::cout << "UNSWAPPED: first deviation ch=" << firstBadCh << " row=" << firstBadRow
                              << " got=" << firstBadGot << " want=" << firstBadWant << std::endl;
                std::cout << "UNSWAPPED ch0 rows 1..8:";
                for (size_t j = 1; j <= 8 && j < nRows; ++j)
                    std::cout << " " << (int)(unsigned short) bswap(data.get<short>(j, 0));
                std::cout << std::endl;

                // All channels equal within a row - SAMPLE ROWS ONLY (row 0 is the header; including
                // it short-circuits this check and says nothing about the samples).
                bool rowConst = true; size_t firstUneq = 0;
                for (size_t j = 1; j < nRows && rowConst; ++j)
                    for (size_t ch = 1; ch < nChannels; ++ch)
                        if (data.get<short>(j, ch) != data.get<short>(j, 0)) {
                            rowConst = false; firstUneq = j; break;
                        }
                std::cout << "SAMPLE ROWS (1..) all channels equal within row: "
                          << (rowConst ? "YES" : "NO");
                if (!rowConst) std::cout << "  (first unequal row " << firstUneq << ")";
                std::cout << std::endl;

                // Persist the raw frame so any further analysis needs no board time.
                std::ofstream raw("frame0.bin", std::ios::binary);
                raw.write(reinterpret_cast<const char *>(ptr->getData().get<short>()),
                          (std::streamsize) ptr->getSize());
                raw.close();
                std::cout << "Wrote frame0.bin (" << ptr->getSize() << " bytes)" << std::endl;
                ptr->release();
            } catch (const std::exception &e) {
                std::cout << "Callback exception: " << e.what() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                frameReceived = true;
            }
            cv.notify_one();
        };

        OnOverflowCallback overflowCallback = [&]() {
            std::cout << "Data overflow occurred!" << std::endl;
            {
                std::lock_guard<std::mutex> lock(mutex);
                overflowed = true;
            }
            cv.notify_one();
        };

        auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
        buffer->registerOnNewDataCallback(callback);
        buffer->registerOnOverflowCallback(overflowCallback);

        session->startScheme();
        bool signalled = false;
        {
            // Bounded wait: if no frame arrives we must still reach stopScheme() rather than be
            // killed mid-acquisition, which would leave the sequencer triggering on the board.
            std::unique_lock<std::mutex> lock(mutex);
            signalled = cv.wait_for(lock, std::chrono::seconds(30),
                                    [&] { return frameReceived || overflowed; });
        }
        session->stopScheme();

        if (!signalled) {
            std::cout << "RESULT: TIMEOUT - no frame within 30 s; scheme stopped cleanly."
                      << std::endl;
            return 1;
        }
        if (overflowed) {
            std::cout << "RESULT: OVERFLOW - capture is not trustworthy." << std::endl;
            return 1;
        }
        std::cout << "RESULT: frame received." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
