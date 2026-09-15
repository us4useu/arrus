#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <regex>

#include <boost/algorithm/string.hpp>
#include <vector>

#include "BlockingQueue.h"
#include "TxTimeoutRegister.h"
#include "Us4REvent.h"
#include "Us4RSubsequence.h"
#include "arrus/common/asserts.h"
#include "arrus/common/cache.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/BlockingQueue.h"
#include "arrus/core/devices/us4r/Us4OEMDataTransferRegistrar.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplane.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplier.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/utils.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;
    using DelayProfiles = std::vector<::arrus::framework::NdArray>;

    enum class State { START_IN_PROGRESS, STARTED, STOP_IN_PROGRESS, STOPPED };

    static float getRxDelay(const ::arrus::ops::us4r::TxRx &op);

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::vector<ProbeSettings> probeSettings,
             ProbeAdapterSettings probeAdapterSettings, std::vector<HighVoltageSupplier::Handle> hv,
             const RxSettings &rxSettings, std::vector<std::unordered_set<unsigned short>> channelsMask,
             std::optional<DigitalBackplane::Handle> backplane, std::vector<Bitstream> bitstreams,
             bool hasIOBitstreamAddressing, const us4r::IOSettings &ioSettings, bool isExternalTrigger,
             bool maskDVDDInterrupt, std::optional<HVPSFuseSettings> hvpsFusetSettings);

    Us4RImpl(Us4RImpl const &) = delete;

    Us4RImpl(Us4RImpl const &&) = delete;

    Device::RawHandle getDevice(const std::string &path) override {
        auto [root, tail] = getPathRoot(path);
        boost::algorithm::trim(root);
        boost::algorithm::trim(tail);
        if (!tail.empty()) {
            throw IllegalArgumentException(arrus::format(
                "Us4R devices allows access only to the top-level devices (got relative path: '{}')", path));
        }
        DeviceId componentId = DeviceId::parse(root);
        return getDevice(componentId);
    }

    Device::RawHandle getDevice(const DeviceId &deviceId) {
        auto ordinal = deviceId.getOrdinal();
        switch (deviceId.getDeviceType()) {
        case DeviceType::Us4OEM: return getUs4OEM(ordinal);
        default: throw DeviceNotFoundException(deviceId);
        }
    }

    Us4OEM::RawHandle getUs4OEM(Ordinal ordinal) override {
        if (ordinal >= us4oems.size()) {
            throw DeviceNotFoundException(DeviceId(DeviceType::Us4OEM, ordinal));
        }
        return us4oems.at(ordinal).get();
    }

    bool isUs4OEMPlus() {
        return this->getMasterOEM()->getDescriptor().isUs4OEMPlus();
    }

    std::pair<Buffer::SharedHandle, std::vector<session::Metadata::SharedHandle>>
    upload(const ops::us4r::Scheme &scheme) override;

    void start() override;

    void stop() override;

    void trigger(bool sync, std::optional<long long> timeout) override;

    void sync(std::optional<long long> timeout) override;

    void setVoltage(Voltage voltage) override;
    void setVoltage(const std::vector<HVVoltage> &voltages) override;

    void disableHV() override;
    void cleanupBuffers(bool cleanupSequencerTransfers = false);

    void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic, bool clip) override;
    void setTgcCurve(const std::vector<float> &x, const std::vector<float> &y, bool applyCharacteristic, bool clip) override;

    void setTgcCurve(const std::vector<float> &tgcCurvePoints) override;
    std::vector<float> getTgcCurvePoints(float endSample) const override;

    void setRxSettings(const RxSettings &settings) override;
    void setPgaGain(uint16 value) override;
    uint16 getPgaGain() override;
    void setLnaGain(uint16 value) override;
    uint16 getLnaGain() override;
    void setLpfCutoff(uint32 value) override;
    void setDtgcAttenuation(std::optional<uint16> value) override;
    void setActiveTermination(std::optional<uint16> value) override;
    uint8_t getNumberOfUs4OEMs() override;
    void setTestPattern(Us4OEM::RxTestPattern pattern) override;
    float getSamplingFrequency() const override;
    float getCurrentSamplingFrequency() const override;
    void checkState() const override;
    void checkVoltage(Voltage voltageMinus, Voltage voltagePlus, float tolerance, int retries, HVModelId hvModel, bool isHVPS);
    unsigned char getVoltage() override;
    float getMeasuredPVoltage() override;
    float getMeasuredMVoltage() override;
    float getMeasuredHVPVoltage(uint8_t oemId) override;
    float getMeasuredHVMVoltage(uint8_t oemId) override;
    void setStopOnOverflow(bool isStopOnOverflow) override;
    bool isStopOnOverflow() const override;
    void setLnaHpfCornerFrequency(uint32_t frequency) override;
    void disableLnaHpf() override;
    void setAdcHpfCornerFrequency(uint32_t frequency) override;
    void setHpfCornerFrequency(uint32_t frequency) override;
    void disableAdcHpf() override;
    void disableAllHpf() override;

    uint16_t getAfe(uint8_t reg) override;
    void setAfe(uint8_t reg, uint16_t val) override;

    void registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const std::vector<Us4OEMBuffer> &srcBuffers,
                              arrus::ops::us4r::Scheme::WorkMode workMode);
    void unregisterOutputBuffer(bool cleanSequencer);
    const char *getBackplaneSerialNumber() override;
    const char *getBackplaneRevision() override;
    const char *getBackplaneFirmwareVersion() override;

    void setParameters(const Parameters &parameters) override;
    void setIOBitstream(BitstreamId id, const std::vector<uint8_t> &levels,
                        const std::vector<uint16_t> &periods) override;
    std::vector<std::vector<uint8_t>> getOEMMappings() const;
    std::optional<Ordinal> getFrameMetadataOEM(const us4r::IOSettings &settings);

    std::vector<unsigned short> getChannelsMask(Ordinal probeNumber) override;
    int getNumberOfProbes() const override;

    Probe::RawHandle getProbe(Ordinal ordinal) override {
        return probes.at(ordinal).get();
    }

    std::pair<std::shared_ptr<framework::Buffer>, std::vector<std::shared_ptr<session::Metadata>>>
    setSubsequences(const std::vector<Slice> &slices, const std::vector<std::optional<float>> &sris) override;

    void setMaximumPulseLength(std::optional<float> maxLength) override;
    float getActualTxFrequency(float frequency) override;
    std::string getDescription() const override;
    float getMinimumTGCValue() const override;

    /**
     * Returns maximum available TGC value, according to the currently set parameters.
     */
    float getMaximumTGCValue() const override;

    std::pair<float, float> getTGCValueRange() const;
    void setVcat(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic, bool clip) override;
    void setVcat(const std::vector<float> &attenuation) override;
    void setVcat(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic, bool clip) override;
    void disableHpf() override;
    virtual Us4OEM::Variant getVariant() override;
    std::vector<int64_t> getHVPSTuningInfo();

private:
    struct VoltageLogbook {
        enum class Polarity { MINUS, PLUS };

        std::string name;
        float voltage;
        Polarity polarity;
    };
    std::vector<VoltageLogbook> logVoltages(HVModelId hvModel, bool isOEMPlus);

    void stopDevice();

    std::tuple<
        std::vector<Us4OEMBuffer>,
        std::vector<FrameChannelMappingImpl::Handle>,
        float,
        std::vector<LogicalToPhysicalOp>,
        std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>>
    >
    uploadSequences(const std::vector<ops::us4r::TxRxSequence> &sequences, uint16_t bufferSize,
                    ops::us4r::Scheme::WorkMode workMode, const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                    const std::vector<framework::NdArray> &txDelayProfiles);
    us4r::TxRxParameters createBitstreamSequenceSelectPreamble(const ops::us4r::TxRxSequence &sequence);
    std::vector<us4r::TxRxParametersSequence>
    convertToInternalSequences(
        const std::vector<ops::us4r::TxRxSequence> &sequences,
        const TxTimeoutRegister &timeoutRegister,
        const std::vector<std::vector<float>> &rxDelays
    );

    /**
     * Applies a given function on all functions.
     * If there was some exception thrown on execution of a given function,
     * an appropriate logging message will printed out, and the result exception,
     * TODO consider implementing rollback mechanism?
     */
    void applyForAllUs4OEMs(const std::function<void(Us4OEM *us4oem)> &func, const std::string &funcName);
    void disableAfeDemod();
    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients);

    void registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                              Us4OEMImplBase::RawHandle us4oem, ::arrus::ops::us4r::Scheme::WorkMode workMode);
    size_t getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const;

    std::function<void()> createReleaseCallback(::arrus::ops::us4r::Scheme::WorkMode workMode, uint16 startFiring,
                                                uint16 stopFiring, bool lastOfLap = true);
    std::function<void()> createOnReceiveOverflowCallback(::arrus::ops::us4r::Scheme::WorkMode workMode,
                                                          Us4ROutputBuffer *buffer, bool isMaster,
                                                          const std::vector<std::pair<uint16, uint16>> &firings);
    std::function<void()> createOnTransferOverflowCallback(::arrus::ops::us4r::Scheme::WorkMode workMode,
                                                           Us4ROutputBuffer *buffer, bool isMaster,
                                                           const std::vector<std::pair<uint16, uint16>> &firings);

    BitstreamId addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods);
    Us4OEMImplBase::RawHandle getMasterOEM() const { return this->us4oems[0].get(); }

    // HS-resume gate baselines (ARRUS_HOST_HS_RESUME=2): the interrupt controller's per-IRQ hardware
    // counters for int5 (HS1/receive stop) and int6 (HS2/transfer stop), per OEM, as read at scheme
    // start. Monotonic and never decremented, so "advanced since the last release" is an exact
    // per-bit "a stop is pending on THIS handshake" - the only signal that separates a stop from an
    // ordinary park, which reads identically in sequencer STATUS.
    std::vector<uint32_t> hsRxIrqBase, hsTxIrqBase;
    // Resume pulses actually issued by the release callback (both arms), logged at stop: the arm
    // is labelled by this number, not by the switch value.
    std::atomic<uint32_t> hsTxPulses{0}, hsRxPulses{0};

    /**
     * Releases the HOST/SYNC park (BLOCK_CLR). Default: the MASTER only, as mainline ARRUS does over
     * PCIe. Under the default HOST scheme (ARRUS_HOST_PARK=element) every board carries a park on
     * every element, but a slave's trigger input is gated by HW_TRIGGER_EN alone and never by its
     * park (RTL, 2026-09-15), so the slaves follow the master's trigger regardless and the master
     * strobe is sufficient; under ARRUS_HOST_PARK=last only the master parks at all.
     * ARRUS_SYNC_ALL_OEMS=1 strobes every board, slaves first (diagnostic; the 2026-09-11 measurement
     * that motivated it was taken in SYNC with the HS stop bits on, where the slave was stopped on a
     * handshake, not parked). ARRUS_SYNC_MASTER_ONLY=1 forces the default path ahead of the others;
     * ARRUS_SYNC_PARK_DELAY_US adds a slave-to-master gap; ARRUS_SYNC_SKIP_RELEASE_CLR=1 issues no
     * release at all.
     *
     * DO NOT confuse this with TriggerStart/TriggerStop, which are master-only for a different
     * reason: there is one trigger generator and its trigger_out feeds every module's trigger_in.
     */
    void syncTriggerAllOEMs() {
        // DIAGNOSTIC, default off: ARRUS_SYNC_PARK_DELAY_US delays the release. The open question
        // (2026-09-11) is whether a BLOCK_CLR issued BEFORE a board has parked is lost, which would
        // make two-board HOST race - the release fires on element completion, i.e. when both boards
        // have DELIVERED, which is not the same event as both having PARKED. A delay is NOT the fix;
        // it only tests the hypothesis. If a delay removes the stall, the real fix is to wait for the
        // park rather than to sleep.
        static const long parkDelayUs = [] {
            const char *v = std::getenv("ARRUS_SYNC_PARK_DELAY_US");
            return v != nullptr ? std::strtol(v, nullptr, 10) : 0L;
        }();
        // ARRUS_SYNC_MASTER_ONLY=1 restores the pre-2026-09-11 master-only behaviour, to isolate
        // whether the per-board change caused a given symptom. Diagnostic only: master-only is
        // WRONG on a multi-board system (BLOCK_CLR is per-sequencer), so this must not be used to
        // "fix" anything - only to attribute.
        static const bool masterOnly = [] {
            const char *v = std::getenv("ARRUS_SYNC_MASTER_ONLY");
            return v != nullptr && v[0] == '1';
        }();
        if (masterOnly) {
            getMasterOEM()->syncTrigger();
            return;
        }
        // The delay, when set, goes BETWEEN the slaves' releases and the master's - that is the
        // ordering under test. The master is the board that resumes GENERATING, so if its BLOCK_CLR
        // takes effect before a slave's has arrived, it can fire a trigger that slave is not yet
        // listening for and that slave falls permanently behind. Releasing master last orders the
        // WRITES; it does not order their ARRIVAL, since each is a separate ~0.5 ms ECB round trip
        // on a different NIC. A delay here is a DIAGNOSTIC: if it removes the drift, the fix is to
        // confirm each slave has resumed before releasing the master, not to sleep.
        // PCIe parity: the release goes to the MASTER ONLY, because only the master parks.
        // AriusConsole/StreamingTest.cpp's mode=="host" loop releases with
        // _us4oem[0]->TriggerSync() - one call, board 0 - and that is consistent with its one park
        // on board 0's last entry. Releasing every board was correct only while every board parked;
        // it reverts together with that, and fixing one without the other would leave N releases
        // chasing 1 park.
        // (The per-board path is kept below under ARRUS_SYNC_ALL_OEMS for A/B against this one.)
        // ARRUS_SYNC_SKIP_RELEASE_CLR=1 (DIAGNOSTIC): issue NO BLOCK_CLR from the release path at
        // all. Tests the hypothesis that a BLOCK_CLR landing on a sequencer that is NOT parked -
        // which under PCIe-style last-entry parking is every release until the final one - disturbs
        // the in-flight egress. If, with this set, all N frames of the first lap arrive and the
        // sequencer parks cleanly at the last entry with BUSY=0, that hypothesis is confirmed.
        static const bool skipReleaseClr = [] {
            const char *v = std::getenv("ARRUS_SYNC_SKIP_RELEASE_CLR");
            return v != nullptr && v[0] == '1';
        }();
        if (skipReleaseClr) {
            return;
        }
        static const bool releaseAllOEMs = [] {
            const char *v = std::getenv("ARRUS_SYNC_ALL_OEMS");
            return v != nullptr && v[0] == '1';
        }();
        if (!releaseAllOEMs) {
            getMasterOEM()->syncTrigger();
            return;
        }
        for (int i = (int) us4oems.size() - 1; i >= 1; --i) {
            us4oems[i]->syncTrigger();
        }
        if (parkDelayUs > 0 && us4oems.size() > 1) {
            std::this_thread::sleep_for(std::chrono::microseconds(parkDelayUs));
        }
        us4oems[0]->syncTrigger();
    }
    std::vector<float> interpolateToSystemTGC(const std::vector<float> &t, const std::vector<float> &y) const;
    void handlePulserInterrupt();

    /**
     * Sets voltage in a safe way, i.e. stops TX/RX sequence before changing the voltage.
     */
    void setVoltage(const std::vector<std::optional<HVVoltage>> &voltages);

    /**
     * Sets voltage without stopping TX/RX sequence.
     * Consider using this method only in case the performance is critical; in other cases, please use setVoltage.
     */
    void setVoltageUnsafe(const std::vector<std::optional<HVVoltage>> &voltages);

    void prepareHostBuffer(unsigned hostBufNElements, ::arrus::ops::us4r::Scheme::WorkMode workMode, std::vector<Us4OEMBuffer> buffers,
                           bool cleanupSequencerTransfers = false);
    std::vector<arrus::session::Metadata::SharedHandle>
    createMetadata(std::vector<FrameChannelMappingImpl::Handle> fcms, float rxTimeOffset) const;

    /**
     * Returns a map sequence id -> list of TX delay arrays.
     */
    std::unordered_map<std::string, DelayProfiles>
    groupTxDelaysBySequence(const std::vector<::arrus::ops::us4r::TxRxSequence> &sequences,
                            const std::vector<::arrus::framework::NdArray> &txDelayProfiles);

    std::tuple<std::string, std::string, size_t> parseTxDelaysConstantName(const std::string &name) const;
    std::tuple<std::string, std::string> parseTxDelaysParamName(const std::string &name) const;
    std::vector<std::vector<float>> getRxDelays(const std::vector<arrus::ops::us4r::TxRxSequence> &seqs);
    std::unordered_map<std::string, SequenceId> getSequenceNameToOrdinalMap(const arrus::ops::us4r::Scheme& scheme) const;
    std::function<void()> createReceiveReleaseCallback(ops::us4r::Scheme::WorkMode workMode, uint16 startFiring, uint16 endFiring);
    /** Sets fuse settings on all us4R OEMs. */
    void setHVPSFuseSettings(const std::optional<HVPSFuseSettings> &settings);

    std::recursive_mutex deviceStateMutex;
    std::mutex triggerMutex;
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<DigitalBackplane::Handle> digitalBackplane;
    std::vector<HighVoltageSupplier::Handle> hv;
    // Settings.
    State state{State::STOPPED};
    // AFE parameters.
    std::mutex afeParamsMutex;
    std::optional<RxSettings> rxSettings;
    std::vector<Probe::Handle> probes;
    std::vector<ProbeSettings> probeSettings;
    ProbeAdapterSettings probeAdapterSettings;
    std::vector<std::unordered_set<ChannelIdx>> channelsMask;
    bool stopOnOverflow{true};
    // Buffers.
    std::vector<Us4OEMBuffer> oemBuffers;
    std::shared_ptr<Us4ROutputBuffer> buffer;
    // HOST-mode stall watchdog (ARRUS_HOST_STALL_MS): releases an element the transport never
    // completed so the boards continue instead of parking forever. See startStallWatchdog().
    bool hostModeScheme{false};
    size_t hostBufferRepeats{1};
    double elementPeriodUs{0.0};
    std::thread stallWatchdog;
    std::atomic<bool> stallWatchdogRun{false};
    void startStallWatchdog();
    void stopStallWatchdog();
    std::vector<std::shared_ptr<Us4OEMDataTransferRegistrar>> transferRegistrar;
    // Other.
    std::vector<Bitstream> bitstreams;
    bool hasIOBitstreamAdressing{false};
    std::optional<Ordinal> frameMetadataOEM{Ordinal(0)};
    bool isExternalTrigger;
    bool maskDVDDInterrupt;

    std::optional<Us4RSubsequenceFactory> subsequenceFactory;
    std::optional<std::vector<Us4RSubsequence>> currentSubsequenceParams;
    /** The currently uploaded scheme */
    std::optional<::arrus::ops::us4r::Scheme> currentScheme;
    std::optional<float> currentRxTimeOffset;
    /** Expected constant name: /SequenceName/parameterName:ordinal */
    const std::regex CONSTANT_NAME_PATTERN{R"(^/([A-Za-z][A-Za-z0-9_:]*)/([^/]+):([0-9]+)$)"};
    /** Expected parameter name: /SequenceName/parameterName */
    const std::regex PARAMETER_NAME_PATTERN{R"(^/([A-Za-z][A-Za-z0-9_:]*)/([^/]+)$)"};
    /** TX/RX sequence name to ordinal number (i.e. position in the list of sequences of the Scheme). */
    std::unordered_map<std::string, SequenceId> sequenceNameToOrdinalMap;
    /** The number of TX delay profiles set for the sequence with the given name */
    std::unordered_map<std::string, size_t> sequenceNumberOfTxDelayProfiles;
    /** HVPS fuse settings. Optional; std::nullopt means that the default HVPS fuse settings will be used
     * (determined in the us4r-api/Us4OEM project */
    std::optional<HVPSFuseSettings> hvpsFuseSettings;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
