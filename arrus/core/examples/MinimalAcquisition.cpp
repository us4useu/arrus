// Smallest ARRUS acquisition: 1 TX/RX event, stops after the first frame.
// Derived from arrus/core/examples/PwiExample.cpp, reduced for a bring-up bench.
//
// SCOPE: this is a BENCH DIAGNOSTIC for the Ethernet/Holoscan port, not a general example.
// It assumes no physical probe, and the frame-inspection block below
// is tied to the CURRENT, UNFINISHED state of the Ethernet data path (see FRAME LAYOUT).
// Once us4r-api strips the header and normalises byte order, that block must be revisited.
//
// HV IS NOT OPTIONAL ON EVERY BOARD - THIS FILE USED TO CLAIM IT WAS. Measured 2026-09-11: with
// HV absent, the STANDARD/AFE58JD18 boards' eight STHV pulsers ALL assert HVM0|HVM1 reference-
// voltage faults (status 0x1028 = 0x0C00). Those INT outputs are open-collector and wired-OR
// onto tx_int, which Sequencer_IP_oemplus.vhd:987 uses to gate the trigger generator - so no
// hardware trigger is ever produced. The symptom is NOT an error: you get exactly one frame
// (PRELOAD fires one acquisition, bypassing the trigger process) and then a silent stall.
// Setting HV clears all eight to 0x0 and the sequence runs. The HF/AFE58JD48 boards do NOT
// fault with HV off - they ran continuous acquisition on this bench for a week - so whether
// this example works without HV is a property of the BOARD VARIANT, not of the example.
//
// Differences from PwiExample that matter on a bring-up bench:
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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <map>
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

        // NOTE: no setVoltage() - there is no hv: block in this configuration. On STANDARD/JD18
        // boards that leaves the pulsers asserting a fault and NO hardware trigger is generated:
        // one frame, then a silent stall. See the HV note at the top of this file.

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
        // MIN_MODE=MANUAL drives the acquisition with SOFTWARE triggers (session->run) instead of
        // the board's hardware trigger generator. On ARIUS_std_seed2 the generator emits nothing -
        // armed, correctly programmed, silent - so this separates "the generator is broken" from
        // "nothing downstream of it works", and if it succeeds it is a usable acquisition path.
        const char *minMode = std::getenv("MIN_MODE");
        const bool manualMode = minMode != nullptr && std::string(minMode) == "MANUAL";
        Scheme scheme(seq, 2, outputBuffer,
                      manualMode ? Scheme::WorkMode::MANUAL : Scheme::WorkMode::HOST);

        auto result = session->upload(scheme);

        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<int> nFrames{0};
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
                // BYTE ORDER: us4OEM DDR4 holds samples big-endian (JESD-native). Over PCIe the
                // Altera DMA IP swapped to little-endian in hardware; over Ethernet the swap is done
                // either by the bitstream (announced in status bit 11) or by us4r-api in software.
                // This check is deliberately agnostic about which side did the swap: it tests the
                // NATIVE read first and only falls back to a byte-swapped read, then reports which
                // orientation produced a clean ramp. That way it stays honest against either an
                // older library (no swap) or a fixed one, and cannot double-swap.
                auto bswap = [](short v) -> short {
                    unsigned short u = (unsigned short) v;
                    return (short) (unsigned short) ((u << 8) | (u >> 8));
                };
                std::cout << "HEADER row 0 (raw int16 x" << nChannels << "):";
                for (size_t ch = 0; ch < nChannels; ++ch) std::cout << " " << data.get<short>(0, ch);
                std::cout << std::endl;

                // Score both orientations over rows 1..N-1: how many channels form a clean +1 ramp.
                // RAMP STRIDE IS NOT ALWAYS 1. It depends on the AFE part and the bitstream
                // variant: an AFE58JD48 on the HF build steps by 1, an AFE58JD18 on the STANDARD
                // build steps by 4. Hardcoding +1 reports a perfectly good capture as corrupt,
                // which it did on 2026-09-08 and cost a false alarm. Derive the stride from the
                // data (modal delta on channel 0) and then demand every channel match it - that
                // still fails on genuinely corrupt data, because corruption has no modal delta
                // that all 32 channels agree on.
                auto modalStride = [&](bool swapped) -> int {
                    std::map<int, long> hist;
                    for (size_t j = 2; j < nRows; ++j) {
                        short a = data.get<short>(j - 1, 0), b = data.get<short>(j, 0);
                        if (swapped) { a = bswap(a); b = bswap(b); }
                        ++hist[(((int) (unsigned short) b) - ((int) (unsigned short) a)) & 0xFFFF];
                    }
                    int best = 1; long bestN = -1;
                    for (const auto &kv : hist) if (kv.second > bestN) { bestN = kv.second; best = kv.first; }
                    return best;
                };
                auto scoreRamp = [&](bool swapped, int stride, size_t &firstBadCh, size_t &firstBadRow,
                                     int &firstBadGot, int &firstBadWant, long &deviations) {
                    size_t clean = 0;
                    bool haveBad = false;
                    deviations = 0;
                    for (size_t ch = 0; ch < nChannels; ++ch) {
                        long badHere = 0;
                        for (size_t j = 2; j < nRows; ++j) {
                            short a = data.get<short>(j - 1, ch), b = data.get<short>(j, ch);
                            if (swapped) { a = bswap(a); b = bswap(b); }
                            int want = (((int) (unsigned short) a) + stride) & 0xFFFF;
                            int cur = (int) (unsigned short) b;
                            if (cur != want) {
                                ++badHere;
                                if (!haveBad) {
                                    haveBad = true;
                                    firstBadCh = ch; firstBadRow = j;
                                    firstBadGot = cur; firstBadWant = want;
                                }
                            }
                        }
                        if (badHere == 0) ++clean; else deviations += badHere;
                    }
                    return clean;
                };
                size_t nbCh = 0, nbRow = 0, sbCh = 0, sbRow = 0;
                int nbGot = 0, nbWant = 0, sbGot = 0, sbWant = 0;
                long nativeDev = 0, swappedDev = 0;
                const int nativeStride = modalStride(false), swappedStride = modalStride(true);
                size_t nativeClean = scoreRamp(false, nativeStride, nbCh, nbRow, nbGot, nbWant, nativeDev);
                size_t swappedClean = scoreRamp(true, swappedStride, sbCh, sbRow, sbGot, sbWant, swappedDev);

                std::cout << "RAMP native  : " << nativeClean << "/" << nChannels
                          << " channels clean, " << nativeDev << " deviations"
                          << " (stride " << nativeStride << ")" << std::endl;
                std::cout << "RAMP swapped : " << swappedClean << "/" << nChannels
                          << " channels clean, " << swappedDev << " deviations"
                          << " (stride " << swappedStride << ")" << std::endl;
                if (nativeClean == nChannels) {
                    std::cout << "VERDICT: samples are correct as delivered (native little-endian)."
                              << std::endl;
                } else if (swappedClean == nChannels) {
                    std::cout << "VERDICT: samples need a 16-bit byte swap - the delivered buffer is "
                                 "big-endian." << std::endl;
                } else {
                    std::cout << "VERDICT: NEITHER orientation yields a clean ramp - payload is not "
                                 "an intact ramp." << std::endl;
                    std::cout << "  native  first deviation ch=" << nbCh << " row=" << nbRow
                              << " got=" << nbGot << " want=" << nbWant << std::endl;
                    std::cout << "  swapped first deviation ch=" << sbCh << " row=" << sbRow
                              << " got=" << sbGot << " want=" << sbWant << std::endl;
                }
                std::cout << "ch0 rows 1..8 native :";
                for (size_t j = 1; j <= 8 && j < nRows; ++j)
                    std::cout << " " << (int) (unsigned short) data.get<short>(j, 0);
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
            ++nFrames;
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

        if (manualMode) {
            const int nRuns = 10;
            int issued = 0;
            for (int i = 0; i < nRuns; ++i) {
                try {
                    session->run(true, 2000);
                    ++issued;
                } catch (const std::exception &e) {
                    std::cout << "run " << i << " threw: " << e.what() << std::endl;
                    break;
                }
            }
            const int got = nFrames.load();
            std::cout << "MANUAL: " << issued << "/" << nRuns << " sync runs returned, frames = "
                      << got << std::endl;
            session->stopScheme();
            std::cout << (got >= nRuns ? "RESULT: MANUAL (software) triggering WORKS."
                                       : "RESULT: MANUAL triggering did NOT deliver every frame.")
                      << std::endl;
            return got > 1 ? 0 : 1;
        }

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
