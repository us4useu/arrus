// Sustained-throughput sweep for the us4OEM data path through ARRUS.
//
// SCOPE: bench diagnostic for the Ethernet/Holoscan port. Assumes a session with no HV and no
// physical probe (see us4r_eth_bench.prototxt). Not a general example.
//
// Runs a continuous acquisition in WorkMode::ASYNC - the sequencer free-runs at the programmed
// PRI and an overflow is reported as an error rather than applying backpressure. That is the
// only mode in which the host is not in the trigger loop: HOST and the MANUAL modes handshake
// with the host once per batch (isWaitForSoftMode), so in those the ECB round-trip, not the data
// path, would set the ceiling.
//
// For each requested PRI it uploads a fresh scheme (PRI is baked into every sequencer entry by
// SetTrigger, so it cannot be changed in place), runs for a fixed time, and records frames and
// bytes delivered to the consumer plus whether the overflow callback fired. The callback does
// nothing but count and release; any work in it would become part of the measurement.
//
// Usage:
//   throughput-test <prototxt> <nSamples> <bufferDepth> <secondsPerPoint> <pri_us> [pri_us ...]
//
// nSamples must be a multiple of 64 (Us4OEMTxRxValidator) and within {64, 65472} on OEM+.
// The sample range starts at 1, not 0, so that ScheduleReceive start = sampleTxStart + 1 = 36,
// the ACQ_RX_DELAY measured as correct on this bench (see MinimalAcquisition.cpp).
//
// PRI FLOOR: ARRUS rejects any op whose txrxTime exceeds the PRI. In the default SEQUENTIAL
// reprogramming mode txrxTime = max(minRxTime, nSamples/fs) + reprogrammingTime, i.e. on an
// OEM+ v3 (fs = 120 MHz, 35 us reprogramming, 20 us minRxTime):
//     1024 samples ->  55 us      16384 samples -> ~172 us      65472 samples -> ~581 us
// A rejected PRI is reported as such and the sweep continues.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "arrus/core/api/arrus.h"

namespace {

// Link flaps mid-point corrupt that point's numbers. Read the kernel's carrier-change counter for
// the NIC us4r-api is using (US4R_ETH_NIC, already in the environment for the run). Returns -1 if
// unavailable so the sweep still runs without it.
long readCarrierChanges() {
    const char *nic = std::getenv("US4R_ETH_NIC");
    if (!nic) return -1;
    std::ifstream f(std::string("/sys/class/net/") + nic + "/carrier_changes");
    long v = -1;
    if (f) f >> v;
    return v;
}

struct PointResult {
    unsigned priUs = 0;
    bool rejected = false;
    std::string rejectReason;
    bool overflowed = false;
    double secondsRun = 0.0;
    uint64_t frames = 0;
    uint64_t bytes = 0;
    double secondsToOverflow = -1.0;
    float fpgaTempBefore = 0.0f;
    float fpgaTempAfter = 0.0f;
    // Steady-state (post-burst) tail: frames/bytes accumulated after the pre-armed burst drains.
    uint64_t ssFrames = 0;
    uint64_t ssBytes = 0;
    double ssSeconds = 0.0;
    bool stalled = false;
    double secondsToStall = -1.0;
    uint64_t expectedFrames = 0;
    long carrierBefore = -1;
    long carrierAfter = -1;
};

void printTable(const std::vector<PointResult> &results, size_t bytesPerFrame) {
    std::cout << "\n"
              << std::setw(8) << "PRI[us]" << std::setw(10) << "target/s" << std::setw(12) << "frames"
              << std::setw(10) << "fps" << std::setw(10) << "MB/s" << std::setw(10) << "Gbit/s"
              << std::setw(9) << "run[s]" << std::setw(9) << "deliv%" << std::setw(12) << "FPGA C"
              << "  status\n";
    for (const auto &r : results) {
        std::cout << std::setw(8) << r.priUs;
        double targetFps = 1e6 / (double) r.priUs;
        std::cout << std::setw(10) << std::fixed << std::setprecision(0) << targetFps;
        if (r.rejected) {
            std::cout << std::setw(12) << "-" << std::setw(10) << "-" << std::setw(10) << "-"
                      << std::setw(10) << "-" << std::setw(9) << "-" << std::setw(9) << "-"
                      << std::setw(12) << "-" << "  REJECTED: " << r.rejectReason
                      << "\n";
            continue;
        }
        double fps = r.secondsRun > 0 ? (double) r.frames / r.secondsRun : 0.0;
        double mbps = r.secondsRun > 0 ? (double) r.bytes / r.secondsRun / 1e6 : 0.0;
        double gbit = mbps * 8.0 / 1000.0;
        std::cout << std::setw(12) << r.frames << std::setw(10) << std::setprecision(0) << fps
                  << std::setw(10) << std::setprecision(1) << mbps << std::setw(10) << std::setprecision(2)
                  << gbit << std::setw(9) << std::setprecision(2) << r.secondsRun;
        double deliv = r.expectedFrames ? 100.0 * (double) r.frames / (double) r.expectedFrames : 0.0;
        std::cout << std::setw(8) << std::setprecision(1) << deliv << "%"
                  << std::setw(5) << std::setprecision(0) << r.fpgaTempBefore << "->"
                  << std::setw(4) << r.fpgaTempAfter;
        if (r.carrierBefore >= 0 && r.carrierAfter != r.carrierBefore) {
            std::cout << "  LINK FLAP (carrier_changes " << r.carrierBefore << "->" << r.carrierAfter
                      << ") - point INVALID";
        } else if (r.overflowed) {
            std::cout << "  OVERFLOW after " << std::setprecision(3) << r.secondsToOverflow << " s";
        } else if (r.stalled) {
            std::cout << "  STALL: no frame for 2 s, last at " << std::setprecision(3) << r.secondsToStall
                      << " s (host-side loss / receiver stopped)";
        } else if (deliv < 99.0) {
            std::cout << "  LOSS: delivered < expected without overflow (host-side drop)";
        } else {
            std::cout << "  ok";
        }
        std::cout << "\n";
    }
    std::cout << "\n(frame = " << bytesPerFrame << " bytes; target/s is the sequencer rate implied by PRI;\n"
              << " deliv% = frames delivered / frames the sequencer must have fired in run[s] - in ASYNC the\n"
              << " sequencer is deterministic, so a shortfall WITHOUT an overflow is host-side receiver loss,\n"
              << " which does not fire ARRUS's sequencer-overflow callback.)\n";
}

}// namespace

int main(int argc, char **argv) noexcept {
    using namespace ::arrus::session;
    using namespace ::arrus::devices;
    using namespace ::arrus::ops::us4r;
    using namespace ::arrus::framework;

    if (argc < 7) {
        std::cerr << "usage: " << argv[0]
                  << " <prototxt> <nSamples> <rxDepth> <hostDepth> <secondsPerPoint> <pri_us> [pri_us ...]\n"
                  << "  rxDepth   = us4OEM/receiver ring (Scheme rxBufferSize)\n"
                  << "  hostDepth = ARRUS host FIFO elements; must be a multiple of rxDepth.\n"
                  << "  Set hostDepth > rxDepth to tell an OEM/receiver re-arm stall (caps at rxDepth)\n"
                  << "  apart from a host-buffer recycle stall (caps at hostDepth).\n";
        return 2;
    }
    const std::string cfgPath = argv[1];
    const unsigned nSamples = (unsigned) std::strtoul(argv[2], nullptr, 10);
    const unsigned rxDepth = (unsigned) std::strtoul(argv[3], nullptr, 10);
    const unsigned hostDepth = (unsigned) std::strtoul(argv[4], nullptr, 10);
    const double secondsPerPoint = std::strtod(argv[5], nullptr);
    std::vector<unsigned> pris;
    for (int i = 6; i < argc; ++i) pris.push_back((unsigned) std::strtoul(argv[i], nullptr, 10));

    // Diagnostic hold: when set, a point runs the full window with the scheme left started even if
    // frames stop, so an external observer (NIC counters) can see whether the sequencer keeps
    // producing. Off by default - it defeats the normal early-STALL behaviour.
    const bool holdMode = std::getenv("THROUGHPUT_HOLD") != nullptr;
    // THROUGHPUT_CHECK: ramp-check every Nth delivered frame. This puts work INSIDE the callback, so
    // a checked run is a correctness run, not a clean throughput run. It confirms each sampled frame
    // is an intact per-frame ramp (catches a mis-folded address delivering wrong/garbled bytes). It
    // does NOT catch generation-mixing of two aligned acquisitions - the AFE ramp resets per frame,
    // so two folded frames that overlap can still look like a clean ramp; the receiver's
    // PSN-generation lap detection is what guards that, tested on its side.
    const bool checkMode = std::getenv("THROUGHPUT_CHECK") != nullptr;
    const uint64_t checkEvery = 100;

    try {
        // INFO, not TRACE: at thousands of frames per second TRACE logging would dominate.
        arrus::useDefaultLoggerFactory()->setClogLevel(arrus::LogSeverity::INFO);

        auto settings = ::arrus::io::readSessionSettings(cfgPath);
        auto session = ::arrus::session::createSession(settings);
        auto ultrasound = (::arrus::devices::Us4R *) session->getDevice("/Ultrasound:0");
        auto probe = ultrasound->getProbe(0);
        unsigned nElements = probe->getModel().getNumberOfElements().product();

        // Verifiable content: ARRUS disables the AFE test pattern during init, so enable RAMP
        // explicitly. Costs nothing in data volume or timing but lets a delivered frame be checked
        // for structural integrity - the direct risk after an address-fold change in the receiver.
        ultrasound->setTestPattern(::arrus::devices::Us4OEM::RxTestPattern::RAMP);

        ::arrus::BitMask aperture(nElements, true);
        std::vector<float> delays(nElements, 0.0f);
        Pulse pulse(6e6, 2, false);
        std::pair<::arrus::uint32, ::arrus::uint32> sampleRange{1, 1 + nSamples};
        const size_t bytesPerFrame = (size_t) nSamples * 32 * sizeof(int16_t);

        std::cout << "throughput sweep: nSamples=" << nSamples << " (" << bytesPerFrame
                  << " B/frame), rxDepth=" << rxDepth << ", hostDepth=" << hostDepth << ", "
                  << secondsPerPoint << " s/point, mode=ASYNC\n";

        std::vector<PointResult> results;
        for (unsigned priUs : pris) {
            PointResult res;
            res.priUs = priUs;

            // Thermal guard: sustained high-rate acquisition heats the OEM. Read the FPGA die
            // temperature before every point and stop the whole sweep above 80 C.
            try {
                res.fpgaTempBefore = ultrasound->getUs4OEM(0)->getFPGATemperature();
            } catch (const std::exception &e) {
                std::cout << "PRI " << priUs << " us: pre-run temperature read failed (" << e.what()
                          << ") - skipping point\n";
                res.rejected = true; res.rejectReason = std::string("ECB: ") + e.what();
                results.push_back(res);
                continue;
            }
            if (res.fpgaTempBefore > 80.0f) {
                std::cout << "ABORT: FPGA at " << res.fpgaTempBefore << " C before PRI " << priUs
                          << " us - stopping sweep to let the board cool.\n";
                break;
            }

            std::vector<TxRx> txrxs;
            txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange),
                               (float) priUs * 1e-6f);
            TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI, 1);
            DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, hostDepth};
            Scheme scheme(seq, (::arrus::uint16) rxDepth, outputBuffer, Scheme::WorkMode::ASYNC);

            std::mutex mutex;
            std::condition_variable cv;
            std::atomic<uint64_t> frames{0};
            std::atomic<uint64_t> bytes{0};
            std::atomic<uint64_t> checked{0};
            std::atomic<uint64_t> checkClean{0};
            bool overflowed = false;
            std::chrono::steady_clock::time_point tStart, tOverflow;
            std::atomic<int64_t> lastFrameNs{0};   // steady_clock ns of the most recent frame
            // Snapshot taken once, at burstCutoff into the run, to measure the post-burst tail.
            const double burstCutoff = 2.0;  // s; the pre-armed table drains well within this
            bool snapTaken = false;
            uint64_t snapFrames = 0, snapBytes = 0;
            std::chrono::steady_clock::time_point snapTime;

            UploadResult uploadResult;
            try {
                uploadResult = session->upload(scheme);
            } catch (const std::exception &e) {
                res.rejected = true;
                res.rejectReason = e.what();
                results.push_back(res);
                std::cout << "PRI " << priUs << " us: rejected (" << e.what() << ")\n";
                continue;
            }

            OnNewDataCallback callback = [&](const BufferElement::SharedHandle &ptr) {
                // Count and release. Nothing else - anything here is inside the measurement.
                uint64_t n = frames.fetch_add(1, std::memory_order_relaxed);
                bytes.fetch_add(ptr->getSize(), std::memory_order_relaxed);
                lastFrameNs.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                                  std::memory_order_relaxed);
                if (checkMode && (n % checkEvery) == 0) {
                    // Native little-endian (the receiver restores LE): after the row-0 header, ch0
                    // should be a clean +1 ramp. Check all channels over rows 2..N-1.
                    auto &d = ptr->getData();
                    size_t nr = (size_t) d.getShape()[0], nc = (size_t) d.getShape()[1];
                    bool clean = true;
                    for (size_t ch = 0; ch < nc && clean; ++ch)
                        for (size_t j = 2; j < nr; ++j) {
                            int prev = (int) (unsigned short) d.get<short>(j - 1, ch);
                            int cur = (int) (unsigned short) d.get<short>(j, ch);
                            if (cur != ((prev + 1) & 0xFFFF)) { clean = false; break; }
                        }
                    checked.fetch_add(1, std::memory_order_relaxed);
                    if (clean) checkClean.fetch_add(1, std::memory_order_relaxed);
                }
                ptr->release();
            };
            OnOverflowCallback overflowCallback = [&]() {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    overflowed = true;
                    tOverflow = std::chrono::steady_clock::now();
                }
                cv.notify_one();
            };
            auto buffer = std::static_pointer_cast<DataBuffer>(uploadResult.getBuffer());
            buffer->registerOnNewDataCallback(callback);
            buffer->registerOnOverflowCallback(overflowCallback);

            res.carrierBefore = readCarrierChanges();
            session->startScheme();
            // Window opens only once the sequencer has been told to run: the ECB round-trips inside
            // startScheme (EnableTxRx / EnableSequencer / TriggerStart) produce no frames and would
            // otherwise bias deliv% low.
            tStart = std::chrono::steady_clock::now();
            bool stalled = false;
            std::chrono::steady_clock::time_point tStall;
            {
                // Poll rather than one long wait: a receiver that goes silent after N frames (one-shot
                // behaviour) or drops under load will never fire ARRUS's overflow callback, so detect
                // it here as "frames arrived, then none for 2 s" and end the point early.
                const auto deadline = tStart + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                   std::chrono::duration<double>(secondsPerPoint));
                std::unique_lock<std::mutex> lock(mutex);
                while (!overflowed) {
                    if (cv.wait_until(lock, std::min(deadline, std::chrono::steady_clock::now()
                                                                   + std::chrono::milliseconds(250)),
                                      [&] { return overflowed; })) break;
                    auto now = std::chrono::steady_clock::now();
                    if (!snapTaken &&
                        std::chrono::duration<double>(now - tStart).count() >= burstCutoff) {
                        snapFrames = frames.load(); snapBytes = bytes.load();
                        snapTime = now; snapTaken = true;
                    }
                    if (now >= deadline) break;
                    int64_t last = lastFrameNs.load(std::memory_order_relaxed);
                    if (!holdMode && frames.load() > 0 && last > 0
                        && now.time_since_epoch().count() - last > 2'000'000'000LL) {
                        stalled = true;
                        tStall = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(last));
                        break;
                    }
                }
            }
            auto tEnd = std::chrono::steady_clock::now();
            session->stopScheme();
            res.carrierAfter = readCarrierChanges();
            // An ECB read right after a heavy run can fail (reply lost under load). That must not
            // throw away the point we just measured: record the failure and keep the numbers.
            try {
                res.fpgaTempAfter = ultrasound->getUs4OEM(0)->getFPGATemperature();
            } catch (const std::exception &e) {
                res.fpgaTempAfter = -1.0f;
                std::cout << "  (post-run temperature read failed: " << e.what() << ")\n";
            }

            res.frames = frames.load();
            res.bytes = bytes.load();
            res.overflowed = overflowed;
            res.stalled = stalled;
            auto endPoint = overflowed ? tOverflow : (stalled ? tStall : tEnd);
            res.secondsRun = std::chrono::duration<double>(endPoint - tStart).count();
            res.expectedFrames = (uint64_t) (res.secondsRun * 1e6 / (double) priUs);
            if (snapTaken) {
                res.ssFrames = res.frames - snapFrames;
                res.ssBytes = res.bytes - snapBytes;
                res.ssSeconds = std::chrono::duration<double>(endPoint - snapTime).count();
            }
            if (stalled) res.secondsToStall = std::chrono::duration<double>(tStall - tStart).count();
            if (overflowed) {
                res.secondsToOverflow = std::chrono::duration<double>(tOverflow - tStart).count();
            }
            results.push_back(res);

            std::cout << "PRI " << priUs << " us: " << res.frames << " frames in " << std::fixed
                      << std::setprecision(2) << res.secondsRun << " s"
                      << (res.ssSeconds > 0.1
                            ? ("  [steady " + std::to_string((long) (res.ssFrames / res.ssSeconds))
                               + " fps, "
                               + std::to_string((long) (res.ssBytes / res.ssSeconds / 1e6)) + " MB/s]")
                            : std::string())
                      << (overflowed ? "  OVERFLOW" : (stalled ? "  STALL" : ""))
                      << ((res.carrierBefore >= 0 && res.carrierAfter != res.carrierBefore) ? "  LINK-FLAP" : "")
                      << (checkMode ? ("  content " + std::to_string(checkClean.load()) + "/"
                                          + std::to_string(checked.load()) + " clean")
                                      : std::string())
                      << "  (expected ~" << res.expectedFrames << ")  FPGA " << std::setprecision(1)
                      << res.fpgaTempBefore << "->" << res.fpgaTempAfter << " C\n";
        }

        printTable(results, bytesPerFrame);
    } catch (const std::exception &e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
