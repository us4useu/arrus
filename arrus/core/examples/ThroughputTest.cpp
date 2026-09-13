// Sustained-throughput sweep for the us4OEM data path through ARRUS.
//
// SCOPE: bench diagnostic for the Ethernet/Holoscan port. Assumes no physical probe
// (see us4r_eth_bench.prototxt). Not a general example.
//
// HV IS REQUIRED ON STANDARD/AFE58JD18 BOARDS - this file used to assume otherwise. Measured
// 2026-09-11: with HV absent their eight STHV pulsers all assert HVM0|HVM1 reference-voltage
// faults (0x1028 = 0x0C00); those INT outputs are open-collector and wired-OR onto tx_int,
// which Sequencer_IP_oemplus.vhd:987 uses to gate the trigger generator. No hardware trigger is
// generated, so the run delivers one frame and stalls - silently, with no error. HF/AFE58JD48
// boards do not fault with HV off, which is why this bench ran HV-less for a week.
//
// Defaults to WorkMode::HOST, where the host is in the trigger loop: the sequencer fires the next
// batch only once the host has consumed the previous one, so the ECB round trip sets the ceiling
// (~1500-1660 fps here, independent of frame size) rather than the data path. On the Ethernet
// bench that is the mode that delivers every frame. THROUGHPUT_MODE=ASYNC selects the free-running
// mode instead, which reaches a higher peak but loses the pre-armed lap at every scheme start and
// parks the sequencer above ~2.6 GB/s.
//
// For each requested PRI it uploads a fresh scheme (PRI is baked into every sequencer entry by
// SetTrigger, so it cannot be changed in place), runs for a fixed time, and records frames and
// bytes delivered to the consumer plus whether the overflow callback fired. The callback does
// nothing but count and release; any work in it would become part of the measurement.
//
// Usage:
//   throughput-test <prototxt> <nSamples> <rxDepth> <hostDepth> <secondsPerPoint> <pri_us> [pri_us ...]
//
// Diagnostic switches (environment; all off by default, each documented where it is read):
//   THROUGHPUT_HOLD          keep the scheme running for the full window after frames stop
//   THROUGHPUT_CHECK         ramp-check every 100th frame (correctness run, not a clean throughput run)
//   THROUGHPUT_PLACEMENT=GPU host buffer on GPU:0 (excludes CHECK and KEEP: they read from the host)
//   THROUGHPUT_MODE=ASYNC|SYNC   work mode other than the default HOST
//   THROUGHPUT_PROBE_ADDR=<hex sequencer word index>   poll that register every 100 ms
//   THROUGHPUT_KICK          after a 1 s gap call SyncReceive+SyncTransfer once
//   THROUGHPUT_RESTART[_WAIT=<s>]  after the point, stop and start again without re-uploading
//   THROUGHPUT_KEEP=N        keep the last N frames and print their header row and ramp breaks
//   THROUGHPUT_PARKTRACE     count the WAIT_FOR_SOFT (int3 / SEQ_IRQ_1) park interrupt PER BOARD.
//                            Needs US4R_ETH_CTRL_EVT_MASK to arm event bit 19 (e.g. 0x3fff0000):
//                            the library default 0x18600000 does NOT include it, and an unarmed
//                            mask looks exactly like the interrupt not existing.
//
// nSamples must be a multiple of 64 (Us4OEMTxRxValidator) and within {64, 16384} on OEM+.
// The sample range starts at 1, not 0, so that ScheduleReceive start = sampleTxStart + 1 = 36,
// the ACQ_RX_DELAY measured as correct on this bench (see MinimalAcquisition.cpp).
//
// PRI FLOOR: ARRUS rejects any op whose txrxTime exceeds the PRI. In the default SEQUENTIAL
// reprogramming mode txrxTime = max(minRxTime, nSamples/fs) + reprogrammingTime, i.e. on an
// OEM+ v3 (fs = 120 MHz, 35 us reprogramming, 20 us minRxTime):
//     1024 samples ->  55 us      4096 samples -> ~69 us      16384 samples -> ~172 us
// A rejected PRI is reported as such and the sweep continues.

#include <algorithm>
#include <atomic>
#include <map>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <tuple>
#include <thread>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "arrus/core/api/arrus.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"

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
              << " deliv% = frames delivered / frames the sequencer would fire in run[s] at this PRI. In ASYNC\n"
              << " the sequencer is deterministic, so a shortfall WITHOUT an overflow is host-side receiver\n"
              << " loss; in HOST the host paces the sequencer, so a shortfall simply means the host round\n"
              << " trip is slower than the PRI and deliv% is not a loss figure.)\n";
}

}// namespace

int main(int argc, char **argv) noexcept {
    // Line-buffer stdout: a session-teardown crash (seen as rc=139 after "Closing session") must
    // not take an already-measured point down with it when output is redirected to a file.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
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
    const bool holdMode = std::getenv("THROUGHPUT_HOLD") != nullptr || std::getenv("THROUGHPUT_KICK") != nullptr;
    // THROUGHPUT_CHECK: ramp-check every Nth delivered frame. This puts work INSIDE the callback, so
    // a checked run is a correctness run, not a clean throughput run. It confirms each sampled frame
    // is an intact per-frame ramp (catches a mis-folded address delivering wrong/garbled bytes). It
    // does NOT catch generation-mixing of two aligned acquisitions - the AFE ramp resets per frame,
    // so two folded frames that overlap can still look like a clean ramp; the receiver's
    // PSN-generation lap detection is what guards that, tested on its side.
    const bool checkMode = std::getenv("THROUGHPUT_CHECK") != nullptr;
    // THROUGHPUT_HSDUMP: dump the per-entry HS handshake flags on every board at the stall.
    const bool hsDump = std::getenv("THROUGHPUT_HSDUMP") != nullptr;
    static std::atomic<int> checkStride{0};  // stride the last checked frame actually used
    static std::atomic<int> checkRestarts{0};  // ramp restarts seen = (OEM blocks - 1)
    // THROUGHPUT_PLACEMENT=GPU: allocate the host buffer on GPU:0 (cudaMalloc on a discrete GPU) so an
    // RDMA-capable transport can land frames in VRAM directly. The ramp check reads the buffer from the
    // host, which is not possible on a discrete-GPU allocation, so the two are mutually exclusive.
    const char *placementEnv = std::getenv("THROUGHPUT_PLACEMENT");
    const bool gpuPlacement = placementEnv != nullptr && std::string(placementEnv) == "GPU";
    const unsigned keepN = std::getenv("THROUGHPUT_KEEP") ? (unsigned) std::strtoul(std::getenv("THROUGHPUT_KEEP"), nullptr, 10) : 0;
    if (gpuPlacement && (checkMode || keepN > 0)) {
        std::cerr << "THROUGHPUT_CHECK / THROUGHPUT_KEEP read the buffer from the host, which a GPU-placed buffer "
                     "does not allow; unset one of them.\n";
        return 2;
    }
    const DeviceId placement(gpuPlacement ? DeviceType::GPU : DeviceType::CPU, 0);
    // THROUGHPUT_MODE selects the work mode; default HOST (see the note at the top of the file).
    //   SYNC - the sequencer waits for the host instead of skipping un-released entries (ARRUS
    //          enables wait-on-overflow only in SYNC).
    //   HOST - the host is in the trigger loop: releasing a batch re-arms the entries AND issues
    //          syncTrigger, so the sequencer fires the next batch of rxDepth entries only once the
    //          host has consumed the previous one. The measured rate is then the host round trip,
    //          not the data path - which is the point of measuring it.
    const char *modeEnv = std::getenv("THROUGHPUT_MODE");
    // Default is HOST on this bench: over Ethernet it is the only mode that delivers every frame -
    // no start-of-scheme lap loss and no sequencer park - at the cost of a ~1500-1660 fps ceiling
    // set by the host round trip. ASYNC reaches a higher peak but loses a lap at every scheme start
    // and parks above ~2.6 GB/s; use THROUGHPUT_MODE=ASYNC deliberately when measuring that.
    const std::string modeName = modeEnv != nullptr ? std::string(modeEnv) : std::string("HOST");
    Scheme::WorkMode workMode = Scheme::WorkMode::HOST;
    if (modeName == "SYNC") workMode = Scheme::WorkMode::SYNC;
    else if (modeName == "ASYNC") workMode = Scheme::WorkMode::ASYNC;
    else if (modeName != "HOST") {
        std::cerr << "THROUGHPUT_MODE must be ASYNC, SYNC or HOST (got " << modeName << ")\n";
        return 2;
    }
    // THROUGHPUT_PROBE_ADDR=<hex>: read that sequencer register every 100 ms during each point, from
    // this thread and through the session's own ECB client (never a second client on the bus), and
    // print the time series of value changes plus a histogram of the low nibble (a state code).
    // THROUGHPUT_SWTRIG: pulse IUs4OEM::SWTrigger() on every poll iteration. SWTrigger drives the
    // TOP-LEVEL sw_trigger PIO (qsys 0x2140), which enters the trigger path at the SAME edge
    // detector the sequencer's own TRIGGER_OUT uses - so it traverses the WHOLE external loop:
    // edge detect, 650-cycle stretch, ext_trigger_out, the physical cable, trigger_in, the capture
    // register and the mux. The sequencer's internal soft_trigger CSR bypasses all of that, which
    // is why MANUAL mode working told us nothing about the loop. Frames arriving under SWTRIG means
    // the loop is intact and the fault is the generator alone; none arriving means the loop is
    // broken and the generator is exonerated.
    const bool swTrigMode = std::getenv("THROUGHPUT_SWTRIG") != nullptr;
    long swTrigCount = 0, swTrigErrors = 0;
    const char *probeEnv = std::getenv("THROUGHPUT_PROBE_ADDR");
    const bool probeMode = probeEnv != nullptr;
    const uint32_t probeAddr = probeMode ? (uint32_t) std::strtoul(probeEnv, nullptr, 16) : 0;
    // THROUGHPUT_KICK: once no frame has arrived for 1 s, call SyncReceive()+SyncTransfer() once -
    // what ARRUS's overflow callback would do if the board's overflow event reached the host - and
    // report whether frames resume. Implies holding the point (no early stall exit).
    const bool kickMode = std::getenv("THROUGHPUT_KICK") != nullptr;
    // THROUGHPUT_RESTART: after the point, stop the scheme and start it again WITHOUT re-uploading,
    // then count frames for 3 s. Tells a latched board state (stays silent) from a load-dependent one
    // (runs again once the offered load was removed).
    const bool restartMode = std::getenv("THROUGHPUT_RESTART") != nullptr;
    // THROUGHPUT_KEEP=N (parsed above): keep copies of the last N delivered frames and, after the
    // point, print each one's header row and where its ramp breaks - to inspect the frames just
    // before a board park. The copy runs inside the callback, so a KEEP run is not a clean one.
    // THROUGHPUT_RESTART_WAIT=<s>: idle time between the stop and the restart (default 0).
    const double restartWaitS = std::getenv("THROUGHPUT_RESTART_WAIT") ? std::strtod(std::getenv("THROUGHPUT_RESTART_WAIT"), nullptr) : 0.0;
    const uint64_t checkEvery = 100;

    try {
        // INFO, not TRACE: at thousands of frames per second TRACE logging would dominate.
        // THROUGHPUT_LOG=TRACE|DEBUG overrides for a diagnostic run.
        {
            const char *lv = std::getenv("THROUGHPUT_LOG");
            arrus::LogSeverity sev = arrus::LogSeverity::INFO;
            if (lv && std::string(lv) == "TRACE") sev = arrus::LogSeverity::TRACE;
            else if (lv && std::string(lv) == "DEBUG") sev = arrus::LogSeverity::DEBUG;
            arrus::useDefaultLoggerFactory()->setClogLevel(sev);
        }

        auto settings = ::arrus::io::readSessionSettings(cfgPath);
        auto session = ::arrus::session::createSession(settings);
        auto ultrasound = (::arrus::devices::Us4R *) session->getDevice("/Ultrasound:0");
        auto probe = ultrasound->getProbe(0);
        unsigned nElements = probe->getModel().getNumberOfElements().product();

        // Verifiable content: ARRUS disables the AFE test pattern during init, so enable RAMP
        // explicitly. Costs nothing in data volume or timing but lets a delivered frame be checked
        // for structural integrity - the direct risk after an address-fold change in the receiver.
        ultrasound->setTestPattern(::arrus::devices::Us4OEM::RxTestPattern::RAMP);
        // ...and turn it OFF again on every exit path. Nothing else does: Us4OEMImpl's destructor
        // does not, and the next session's constructor-time DisableTestPatterns() only runs when
        // there is a next session. Without this, every run left AFE_TX_TRIG_MUX_SEL (control bit
        // 12) set and eight AFEs generating a ramp into nothing between sessions - measured
        // 2026-09-13 (control.data 0x1234 after every session, 0x0234 after an explicit disable).
        struct TestPatternOff {
            ::arrus::devices::Us4R *us4r;
            ~TestPatternOff() {
                try { us4r->setTestPattern(::arrus::devices::Us4OEM::RxTestPattern::OFF); }
                catch (const std::exception &e) { std::cerr << "testpat: OFF at exit failed: " << e.what() << "\n"; }
                catch (...) {}
            }
        } testPatternOff{ultrasound};

        // THROUGHPUT_PARKTRACE: does the WAIT_FOR_SOFT park interrupt arrive on EVERY board, or only
        // on the master? A per-board release keyed on the park event would rest entirely on that,
        // and it has only ever been observed on the master - so measure it before anyone designs
        // against it. Registered directly on IUs4OEM because Us4OEMInterrupt (the public ARRUS enum)
        // does not expose WAIT_FOR_SOFT.
        std::vector<std::shared_ptr<std::atomic<long>>> parkCounts, doneCounts;
        const bool parkTrace = std::getenv("THROUGHPUT_PARKTRACE") != nullptr;
        if (parkTrace) {
            const auto nOems = ultrasound->getNumberOfUs4OEMs();
            for (unsigned o = 0; o < nOems; ++o) {
                auto counter = std::make_shared<std::atomic<long>>(0);
                parkCounts.push_back(counter);
                auto doneCounter = std::make_shared<std::atomic<long>>(0);
                doneCounts.push_back(doneCounter);
                try {
                    auto *oemImpl = dynamic_cast<Us4OEMImpl *>(ultrasound->getUs4OEM((::arrus::devices::Ordinal) o));
                    oemImpl->getIUs4OEM()->RegisterCallback(IUs4OEM::MSINumber::WAIT_FOR_SOFT,
                                                            [counter]() { counter->fetch_add(1, std::memory_order_relaxed); });
                    // CONTROL: EVENTDONE (int4) registered the SAME way, at the same time. If this
                    // one counts and WAIT_FOR_SOFT does not, late registration works and int3 is
                    // specifically not being delivered. If NEITHER counts, the registration path
                    // itself is what does not work post-construction, and nothing can be concluded
                    // about int3 at all.
                    oemImpl->getIUs4OEM()->RegisterCallback(IUs4OEM::MSINumber::EVENTDONE,
                                                            [doneCounter]() { doneCounter->fetch_add(1, std::memory_order_relaxed); });
                } catch (const std::exception &e) {
                    std::cout << "parktrace: OEM " << o << " callback registration failed: " << e.what() << std::endl;
                }
            }
            const char *m = std::getenv("US4R_ETH_CTRL_EVT_MASK");
            const char *evt = std::getenv("US4R_ETH_CTRL_EVT");
            std::cout << "parktrace: armed on " << (unsigned) nOems << " OEM(s), US4R_ETH_CTRL_EVT_MASK="
                      << (m ? m : "(unset - event bit 19 is NOT in the 0x18600000 default, expect zero counts)")
                      << ", US4R_ETH_CTRL_EVT=" << (evt ? evt : "(unset - dispatch may be off)")
                      << std::endl;
        }

        // THROUGHPUT_TESTPAT_OFF=1: issue IUs4OEM::DisableTestPatterns() on every OEM and exit
        // without HV, upload or acquisition. One-off board maintenance (2026-09-13, Mateusz's
        // authorisation): a probe left AFE_TX_TRIG_MUX_SEL (control bit 12) set on board 1 and the
        // board has idled at ~83 C since; nothing in an ordinary ARRUS session clears that bit.
        // The read-back of control.data is done from outside (EcbProbe 0x2000), not here.
        if (const char *tp = std::getenv("THROUGHPUT_TESTPAT_OFF"); tp != nullptr && std::string(tp) == "1") {
            for (::arrus::devices::Ordinal o = 0; o < ultrasound->getNumberOfUs4OEMs(); ++o) {
                auto *oemImpl = dynamic_cast<Us4OEMImpl *>(ultrasound->getUs4OEM(o));
                oemImpl->getIUs4OEM()->DisableTestPatterns();
                std::cout << "testpat: DisableTestPatterns() issued on OEM " << (int) o << "\n";
            }
            std::cout << "testpat: done, exiting without acquisition\n";
            return 0;
        }
        // THROUGHPUT_HV=<volts>: energise the OEM's internal HVPS before uploading. REQUIRED on the
        // standard/AFE58JD18 boards for any run that depends on HARDWARE triggers - without HV
        // references the pulsers assert HVM0|HVM1, hold the wired-OR tx_int low, and the trigger
        // generator is gated: you get one frame (PRELOAD) and a silent stall. 10 V was measured
        // sufficient to clear all eight; lower was not explored.
        // Needs a config with an `hv:` block (us4r_eth_bench_hv.prototxt) - setVoltage() throws
        // "No HV have been set." otherwise. Sets TX amplitude 2 only; see that file's header for
        // the strictly-increasing rule if both amplitudes are ever needed.
        if (const char *hvEnv = std::getenv("THROUGHPUT_HV")) {
            const int hvVolts = std::atoi(hvEnv);
            if (hvVolts > 0) {
                try {
                    ultrasound->setVoltage((::arrus::Voltage) hvVolts);
                    std::cout << "hv: set to " << hvVolts << " V" << std::endl;
                } catch (const std::exception &e) {
                    std::cout << "hv: setVoltage(" << hvVolts << ") FAILED: " << e.what()
                              << "  (is the config's hv: block present?)" << std::endl;
                    throw;
                }
            }
        }

        ::arrus::BitMask aperture(nElements, true);
        std::vector<float> delays(nElements, 0.0f);
        Pulse pulse(6e6, 2, false);
        std::pair<::arrus::uint32, ::arrus::uint32> sampleRange{1, 1 + nSamples};
        const size_t bytesPerFrame = (size_t) nSamples * 32 * sizeof(int16_t);

        std::cout << "throughput sweep: nSamples=" << nSamples << " (" << bytesPerFrame
                  << " B/frame), rxDepth=" << rxDepth << ", hostDepth=" << hostDepth << ", "
                  << secondsPerPoint << " s/point, mode=" << modeName << ", placement=" << placement.toString() << "\n";

        std::vector<PointResult> results;
        for (unsigned priUs : pris) {
            PointResult res;
            res.priUs = priUs;

            // Thermal guard: sustained high-rate acquisition heats the OEM. Read the FPGA die
            // temperature before every point and stop the whole sweep above the limit: 80 C by
            // default, THROUGHPUT_TMAX overrides. 80 is this test's own conservative number, not a
            // vendor limit (Arria 10 Tj max is 100 C); the limit in force is printed on abort.
            static const float tMax = [] {
                const char *v = std::getenv("THROUGHPUT_TMAX");
                return v != nullptr ? std::strtof(v, nullptr) : 80.0f;
            }();
            try {
                res.fpgaTempBefore = ultrasound->getUs4OEM(0)->getFPGATemperature();
            } catch (const std::exception &e) {
                std::cout << "PRI " << priUs << " us: pre-run temperature read failed (" << e.what()
                          << ") - skipping point\n";
                res.rejected = true; res.rejectReason = std::string("ECB: ") + e.what();
                results.push_back(res);
                continue;
            }
            if (res.fpgaTempBefore > tMax) {
                std::cout << "ABORT: FPGA at " << res.fpgaTempBefore << " C before PRI " << priUs
                          << " us (limit " << tMax << " C) - stopping sweep to let the board cool.\n";
                break;
            }

            std::vector<TxRx> txrxs;
            txrxs.emplace_back(Tx(aperture, delays, pulse), Rx(aperture, sampleRange),
                               (float) priUs * 1e-6f);
            TxRxSequence seq(txrxs, {}, TxRxSequence::NO_SRI, 1);
            DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, hostDepth, placement};
            Scheme scheme(seq, (::arrus::uint16) rxDepth, outputBuffer, workMode);

            std::mutex mutex;
            std::condition_variable cv;
            std::atomic<uint64_t> frames{0};
            std::atomic<uint64_t> bytes{0};
            std::atomic<uint64_t> checked{0};
            std::atomic<uint64_t> checkClean{0};
            // Deliveries per host-buffer element: a transport that loses whole frames shows whether
            // the loss is uniform or tied to particular slots (e.g. the last slot of the ring).
            std::vector<std::atomic<uint64_t>> slotCount(hostDepth);
            // (t, per-OEM values, frames so far). PER OEM, not just board 0: the whole point on a
            // multi-board scheme is which board's sequencer is in a different state from the other.
            std::vector<std::tuple<double, std::vector<uint32_t>, uint64_t>> probeSamples;
            std::string hsDumpText;
            unsigned probeErrors = 0;
            bool kicked = false; double kickTime = 0; uint64_t kickFrames = 0; std::string kickResult;
            std::atomic<bool> restartPhase{false};
            struct KeptFrame { uint64_t n; size_t pos; std::vector<int16_t> data; size_t rows, chans; };
            std::vector<KeptFrame> kept(keepN);
            std::mutex keptMutex;
            std::atomic<int> firstAfterRestart{-1};
            for (auto &c : slotCount) c.store(0);
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
                if (ptr->getPosition() < slotCount.size()) {
                    slotCount[ptr->getPosition()].fetch_add(1, std::memory_order_relaxed);
                }
                if (keepN > 0) {
                    auto &d = ptr->getData();
                    size_t nr = (size_t) d.getShape()[0], nc = (size_t) d.getShape()[1];
                    std::lock_guard<std::mutex> g(keptMutex);
                    auto &k = kept[n % keepN];
                    k.n = n; k.pos = ptr->getPosition(); k.rows = nr; k.chans = nc;
                    k.data.assign(d.get<int16_t>(), d.get<int16_t>() + nr * nc);
                }
                if (restartPhase.load(std::memory_order_relaxed)) {
                    int expected = -1;
                    firstAfterRestart.compare_exchange_strong(expected, (int) ptr->getPosition());
                }
                lastFrameNs.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                                  std::memory_order_relaxed);
                if (checkMode && (n % checkEvery) == 0) {
                    // Samples arrive little-endian (swapped by the bitstream or by the receiver):
                    // after the row-0 header every channel should be a clean ramp over rows 2..N-1.
                    // DERIVE THE STRIDE, NEVER ASSUME IT. This check hardcoded prev+1 until
                    // 2026-09-11 and reported 0/50 on the standard/AFE58JD18 boards, whose ramp
                    // steps by 4 - a total-corruption verdict on data that was intact. The stride
                    // is a property of the AFE variant, so the only safe reading is the modal
                    // first difference of channel 0 in this very frame.
                    auto &d = ptr->getData();
                    size_t nr = (size_t) d.getShape()[0], nc = (size_t) d.getShape()[1];
                    int stride = 1;
                    if (nr > 3) {
                        std::map<int, long> hist;
                        for (size_t j = 3; j < nr; ++j) {
                            int a = (int) (unsigned short) d.get<short>(j - 1, 0);
                            int b = (int) (unsigned short) d.get<short>(j, 0);
                            ++hist[(b - a) & 0xFFFF];
                        }
                        long bestN = -1;
                        for (const auto &kv : hist) if (kv.second > bestN) { bestN = kv.second; stride = kv.first; }
                    }
                    // A MULTI-OEM ELEMENT IS BLOCKS OF ROWS, ONE PER OEM, AND THE AFE RAMP RESTARTS
                    // AT EACH BOUNDARY. Measured 2026-09-11 on a 2-OEM scheme: 2048 rows, ch0
                    // running 52,56,60,... and ending at 4140 - one OEM's worth of ramp, not two.
                    // A strict "every step matches the stride" test therefore calls a perfectly
                    // intact two-board frame corrupt. Accept a step that RESTARTS (value drops),
                    // and only a few of them; anything else must follow the stride. Random
                    // corruption still fails: it produces forward jumps, or far too many restarts.
                    // COUNT the anomalies per channel, do not fail on the first. A multi-OEM
                    // element is one block of rows per OEM, and each block carries its OWN header
                    // rows and restarts its ramp - so a 2-OEM frame has a handful of legitimate
                    // non-ramp steps in the middle that no amount of stride derivation removes.
                    // Only row 0..1 of the FIRST block is skipped by starting at j=2; the second
                    // block's header sits mid-frame. Budget a few anomalies per boundary and
                    // report the worst channel's count, so the number is visible rather than a
                    // bare verdict. Real corruption is nowhere near this budget: a garbled frame
                    // deviates on hundreds or thousands of rows, not three.
                    const int maxAnomalies = 3;  // ~1 restart + up to 2 header rows per boundary
                    bool clean = stride != 0;
                    int worst = 0;
                    for (size_t ch = 0; ch < nc; ++ch) {
                        int anomalies = 0;
                        for (size_t j = 2; j < nr; ++j) {
                            int prev = (int) (unsigned short) d.get<short>(j - 1, ch);
                            int cur = (int) (unsigned short) d.get<short>(j, ch);
                            if (cur != ((prev + stride) & 0xFFFF)) ++anomalies;
                        }
                        if (anomalies > worst) worst = anomalies;
                    }
                    if (worst > maxAnomalies) clean = false;
                    checkRestarts.store(worst, std::memory_order_relaxed);
                    checked.fetch_add(1, std::memory_order_relaxed);
                    if (clean) checkClean.fetch_add(1, std::memory_order_relaxed);
                    checkStride.store(stride, std::memory_order_relaxed);
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
                                                                   + std::chrono::milliseconds(probeMode ? 100 : 250)),
                                      [&] { return overflowed; })) break;
                    auto now = std::chrono::steady_clock::now();
                    if (probeMode) {
                        lock.unlock();
                        try {
                            std::vector<uint32_t> vs;
                            for (unsigned o = 0; o < ultrasound->getNumberOfUs4OEMs(); ++o) {
                                auto *impl = dynamic_cast<Us4OEMImpl *>(
                                    ultrasound->getUs4OEM((::arrus::devices::Ordinal) o));
                                vs.push_back(impl->getIUs4OEM()->SequencerReadRegister(probeAddr));
                            }
                            probeSamples.emplace_back(std::chrono::duration<double>(now - tStart).count(),
                                                      vs, frames.load());
                        } catch (const std::exception &) {
                            probeErrors++;
                        }
                        lock.lock();
                    }
                    if (!snapTaken &&
                        std::chrono::duration<double>(now - tStart).count() >= burstCutoff) {
                        snapFrames = frames.load(); snapBytes = bytes.load();
                        snapTime = now; snapTaken = true;
                    }
                    if (now >= deadline) break;
                    int64_t last = lastFrameNs.load(std::memory_order_relaxed);
                    if (kickMode && !kicked && frames.load() > 0 && last > 0
                        && now.time_since_epoch().count() - last > 1'000'000'000LL) {
                        kicked = true;
                        kickTime = std::chrono::duration<double>(now - tStart).count();
                        kickFrames = frames.load();
                        lock.unlock();
                        try {
                            auto *impl = dynamic_cast<Us4OEMImpl *>(ultrasound->getUs4OEM(0));
                            impl->getIUs4OEM()->SyncReceive();
                            impl->getIUs4OEM()->SyncTransfer();
                            kickResult = "ok";
                        } catch (const std::exception &e) {
                            kickResult = std::string("threw: ") + e.what();
                        }
                        lock.lock();
                    }
                    if (swTrigMode) {
                        lock.unlock();
                        try {
                            auto *impl = dynamic_cast<Us4OEMImpl *>(ultrasound->getUs4OEM(0));
                            impl->getIUs4OEM()->SWTrigger();
                            ++swTrigCount;
                        } catch (const std::exception &) { ++swTrigErrors; }
                        lock.lock();
                    }
                    if (!holdMode && frames.load() > 0 && last > 0
                        && now.time_since_epoch().count() - last > 2'000'000'000LL) {
                        stalled = true;
                        tStall = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(last));
                        // THROUGHPUT_HSDUMP: read the HS handshake flags ONCE, here, at the moment
                        // the stall is declared. The sequencer SETS each entry's HS bit after
                        // executing it and software must clear it; in HOST mode the release does
                        // that. If the ring has filled because releases are not reaching the board,
                        // the entries read NOT-ready and the board stops feeding it. Read once, not
                        // per poll: 2 calls x nEntries x nOEMs is a lot of ECB round trips and
                        // polling it would perturb the thing being measured.
                        if (hsDump) {
                            lock.unlock();
                            try {
                                for (unsigned o = 0; o < ultrasound->getNumberOfUs4OEMs(); ++o) {
                                    auto *impl = dynamic_cast<Us4OEMImpl *>(
                                        ultrasound->getUs4OEM((::arrus::devices::Ordinal) o));
                                    std::string line = "  hsdump oem" + std::to_string(o) + " entries 0.."
                                                     + std::to_string(rxDepth - 1) + " (R=ready-for-receive, "
                                                       "T=ready-for-transfer, . = NOT ready):";
                                    for (unsigned e = 0; e < rxDepth; ++e) {
                                        bool r = impl->getIUs4OEM()->IsEntryReadyForReceive((uint16_t) e);
                                        bool t = impl->getIUs4OEM()->IsEntryReadyForTransfer((uint16_t) e);
                                        line += std::string(" ") + std::to_string(e) + ":"
                                              + (r ? "R" : ".") + (t ? "T" : ".");
                                    }
                                    hsDumpText += line + "\n";
                                    // Decode the pulser IRQ state on each board at the stall.
                                    // LogPulsersInterruptRegister exists on IUs4OEM and is called
                                    // from NOWHERE in either tree - the same way this morning's
                                    // one-firing stall (eight pulsers asserting HVM0|HVM1, pulling
                                    // the wired-OR tx_int low and gating the trigger generator)
                                    // stayed undiagnosed: the decoder was there and nobody called
                                    // it. A board stuck in transmission with a live pulser fault is
                                    // that same shape. It logs rather than returns, so the output
                                    // lands in the session log, not here.
                                    impl->getIUs4OEM()->LogPulsersInterruptRegister();
                                }
                            } catch (const std::exception &e) {
                                hsDumpText += std::string("  hsdump failed: ") + e.what() + "\n";
                            }
                            lock.lock();
                        }
                        break;
                    }
                }
            }
            auto tEnd = std::chrono::steady_clock::now();
            session->stopScheme();
            std::string restartResult;
            if (restartMode) {
                const uint64_t f0 = frames.load();
                std::vector<uint64_t> slotsBefore;
                for (auto &c : slotCount) slotsBefore.push_back(c.load());
                try {
                    if (restartWaitS > 0) std::this_thread::sleep_for(std::chrono::duration<double>(restartWaitS));
                    restartPhase.store(true);
                    session->startScheme();
                    std::string restartProbe;
                    for (int i = 0; i < 12; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                        if (probeMode) {
                            try {
                                auto *impl = dynamic_cast<Us4OEMImpl *>(ultrasound->getUs4OEM(0));
                                uint32_t v = impl->getIUs4OEM()->SequencerReadRegister(probeAddr);
                                char buf[32]; std::snprintf(buf, sizeof buf, " 0x%x", v); restartProbe += buf;
                            } catch (const std::exception &) { restartProbe += " ERR"; }
                        }
                    }
                    const uint64_t f1 = frames.load();
                    session->stopScheme();
                    std::string slotsDelta;
                    for (size_t i = 0; i < slotCount.size(); ++i)
                        slotsDelta += " " + std::to_string(slotCount[i].load() - slotsBefore[i]);
                    restartResult = std::to_string(f1 - f0) + " frames in 3 s after stop+" + std::to_string((int) restartWaitS) + " s idle+start without re-upload"
                                    + "; first element after restart: " + std::to_string(firstAfterRestart.load())
                                    + "; slots after restart:" + slotsDelta
                                    + (probeMode ? " (probe during restart:" + restartProbe + ")" : std::string());
                } catch (const std::exception &e) {
                    restartResult = std::string("restart threw: ") + e.what();
                }
            }
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
                                          + std::to_string(checked.load()) + " clean (stride "
                                          + std::to_string(checkStride.load()) + ", "
                                          + std::to_string(checkRestarts.load()) + " max anomalies/ch)")
                                      : std::string())
                      << "  (expected ~" << res.expectedFrames << ")  FPGA " << std::setprecision(1)
                      << res.fpgaTempBefore << "->" << res.fpgaTempAfter << " C\n";
            std::cout << "  slots:";
            for (auto &c : slotCount) std::cout << " " << c.load();
            std::cout << "\n";
            if (restartMode) std::cout << "  restart: " << restartResult << "\n";
            if (keepN > 0) {
                std::vector<KeptFrame> ks;
                { std::lock_guard<std::mutex> g(keptMutex); ks = kept; }
                std::sort(ks.begin(), ks.end(), [](const KeptFrame &a, const KeptFrame &b) { return a.n < b.n; });
                for (auto &k : ks) {
                    if (k.data.empty()) continue;
                    std::cout << "  kept frame #" << k.n << " slot " << k.pos << " header:";
                    for (size_t c = 0; c < k.chans; ++c) std::cout << " " << k.data[c];
                    // Ramp breaks on channel 0. DERIVE THE STRIDE - this printed "+1" until
                    // 2026-09-11 and reported 2046 breaks on an intact stride-4 frame. A restart
                    // (value drops) is a per-OEM block boundary, not a break; it is counted
                    // separately so a multi-OEM frame does not read as corrupt.
                    int kStride = 1;
                    if (k.rows > 3) {
                        std::map<int, long> kh;
                        for (size_t r = 3; r < k.rows; ++r) {
                            int a = (uint16_t) k.data[(r - 1) * k.chans], b = (uint16_t) k.data[r * k.chans];
                            ++kh[(b - a) & 0xFFFF];
                        }
                        long bn = -1;
                        for (const auto &kv : kh) if (kv.second > bn) { bn = kv.second; kStride = kv.first; }
                    }
                    std::string breaks; unsigned nb = 0, nrest = 0;
                    for (size_t r = 2; r < k.rows; ++r) {
                        int prev = (uint16_t) k.data[(r - 1) * k.chans], cur = (uint16_t) k.data[r * k.chans];
                        if (cur == ((prev + kStride) & 0xFFFF)) continue;
                        if (cur < prev) { ++nrest; continue; }
                        {
                            if (nb < 6) breaks += " row " + std::to_string(r) + ":" + std::to_string(prev) + "->" + std::to_string(cur);
                            ++nb;
                        }
                    }
                    std::cout << " | ch0 stride=" << kStride << " row1=" << (uint16_t) k.data[k.chans]
                              << " row" << (k.rows - 1) << "=" << (uint16_t) k.data[(k.rows - 1) * k.chans]
                              << " block restarts=" << nrest << " ramp breaks=" << nb << breaks << "\n";
                }
            }
            if (kickMode) {
                std::cout << "  kick: " << (kicked ? "SyncReceive+SyncTransfer at " + std::to_string(kickTime) + " s after "
                                                      + std::to_string(kickFrames) + " frames -> " + kickResult
                                                      + ", frames after kick: " + std::to_string(res.frames - kickFrames)
                                                  : std::string("not needed (no 1 s gap)")) << "\n";
            }
            if (swTrigMode) {
                std::cout << "  swtrig: " << swTrigCount << " SWTrigger() call(s), " << swTrigErrors
                          << " threw; frames this point: " << res.frames << "\n";
            }
            if (!hsDumpText.empty()) { std::cout << hsDumpText; }
            if (parkTrace) {
                std::cout << "  parktrace: WAIT_FOR_SOFT(int3) per OEM:";
                for (size_t o = 0; o < parkCounts.size(); ++o) {
                    std::cout << " oem" << o << "=" << parkCounts[o]->load();
                }
                std::cout << " | CONTROL EVENTDONE(int4):";
                for (size_t o = 0; o < doneCounts.size(); ++o) {
                    std::cout << " oem" << o << "=" << doneCounts[o]->load();
                }
                std::cout << "\n";
            }
            if (probeMode) {
                unsigned hist[16] = {0};
                std::cout << "  probe 0x" << std::hex << probeAddr << std::dec << ": " << probeSamples.size()
                          << " reads, " << probeErrors << " failed; value changes (t s: oem0/oem1/... @frames):";
                std::vector<uint32_t> prev;
                for (auto &smp : probeSamples) {
                    const auto &vs = std::get<1>(smp);
                    if (!vs.empty()) hist[vs[0] & 0xF]++;
                    if (vs != prev) {
                        std::cout << " " << std::fixed << std::setprecision(1) << std::get<0>(smp) << ":";
                        for (size_t o = 0; o < vs.size(); ++o) {
                            if (o) std::cout << " | ";
                            if (probeAddr == 0x50002) {
                                // Sequencer STATUS (sequencer2RegsDef.h / guide 4.37):
                                //   [13:0] CURRENT_INDEX, [14] BUSY, [30:17] LAST_INDEX.
                                // LAST_INDEX is the last entry FULLY executed. CURRENT_INDEX with
                                // BUSY=1 is the entry executing now; with BUSY=0 it is the entry
                                // LOADED and awaiting the next trigger. So last==current with
                                // BUSY=0 means PARKED AND UNRELEASED, while current==last+1 with
                                // BUSY=0 means released and waiting for a trigger that has not come.
                                std::cout << "oem" << o << " last=" << ((vs[o] >> 17) & 0x3FFF)
                                          << " cur=" << (vs[o] & 0x3FFF)
                                          << " busy=" << ((vs[o] >> 14) & 1);
                            } else {
                                std::cout << "oem" << o << " 0x" << std::hex << vs[o] << std::dec;
                            }
                        }
                        std::cout << " @" << std::get<2>(smp);
                        prev = vs;
                    }
                }
                std::cout << "\n  probe low-nibble histogram (oem0):";
                for (int i = 0; i < 16; ++i) if (hist[i]) std::cout << " " << i << ":" << hist[i];
                std::cout << "\n";
            }
        }

        printTable(results, bytesPerFrame);
    } catch (const std::exception &e) {
        std::cerr << "FATAL: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
