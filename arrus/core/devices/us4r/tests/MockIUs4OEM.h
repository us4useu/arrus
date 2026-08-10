#ifndef ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
#define ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <ius4oem.h>

class MockIUs4OEM : public IUs4OEM {
public:
    MOCK_METHOD(unsigned int, getId, (), (override));
    MOCK_METHOD(uint32_t, getFirmwareVersion, (), (override));
    MOCK_METHOD(uint32_t, getTxFirmwareVersion, (), (override));
    MOCK_METHOD(void, checkFirmwareVersion, (), (override));
    MOCK_METHOD(bool, isPowereddown, (), (override));
    MOCK_METHOD(void, initialize, (int), (override));
    MOCK_METHOD(void, synchronize, (), (override));
    MOCK_METHOD(void, scheduleReceive,
            (const size_t firing, const size_t address, const size_t length, const uint32_t start, const uint32_t decimation, const size_t rxMapId, const std::function<void()>& callback),
    (override));
    MOCK_METHOD(void, clearScheduledReceive, (), (override));
    MOCK_METHOD(void, transferRxBufferToHost,
            (unsigned char * dstAddress, size_t length, size_t srcAddress, bool isGpu),
    (override));
    MOCK_METHOD(void, releaseTransferRxBufferToHost,
        (unsigned char * dstAddress, size_t length, size_t srcAddress),
    (override));
    MOCK_METHOD(void, setRxSettings, (const us4us::us4r::RxSettings &rxSettings, bool force), (override));
    MOCK_METHOD(void, initializeTx, (bool ignoreHVPSCalibration), (override));
    MOCK_METHOD(void, setNumberOfFirings, (const unsigned short nFirings),
    (override));
    MOCK_METHOD(::us4us::us4r::Vector<float>, setTxDelays, (const ::us4us::us4r::Span<float> &delays, const uint16_t firing, size_t profile, size_t sequenceId), (override));
    MOCK_METHOD(void, setTxDelays, (const std::vector<size_t> &profiles), (override));
    MOCK_METHOD(float, setTxFreqency,
            (const float frequency, const unsigned short firing),
    (override));
    MOCK_METHOD(uint32_t, setTxHalfPeriods,
            (uint32_t nop, const unsigned short firing), (override));
    MOCK_METHOD(void, setTxInvert, (bool onoff, const unsigned short firing),
    (override));
    MOCK_METHOD(void, setTxCw, (bool onoff, const unsigned short firing),
    (override));
    MOCK_METHOD(void, setRxAperture,
            (const std::bitset<NCH>& aperture, const unsigned short firing),
    (override));
    MOCK_METHOD(void, setTxAperture,
            (const std::bitset<NCH>& aperture, const unsigned short firing),
    (override));
    MOCK_METHOD(void, setActiveChannelGroup,
            (const std::bitset<NCH / 8>& group, const unsigned short firing),
    (override));
    MOCK_METHOD(void, setRxTime,
            (const float time, const unsigned short firing), (override));
    MOCK_METHOD(void, setRxDelay,
            (const float delay, const unsigned short firing), (override));
    MOCK_METHOD(void, enableTxRx, (), (override));
    MOCK_METHOD(void, enableSequencer, (bool txConfOnTrigger, uint16_t startEntry, bool maskDVDDInterrupt), (override));
    MOCK_METHOD(void, setRxChannelMapping,
            ( const std::vector<uint8_t> & mapping, const uint16_t rxMapId),
    (override));
    MOCK_METHOD(void, setTxChannelMapping,
            (const unsigned char srcChannel, const unsigned char dstChannel),
    (override));
    MOCK_METHOD(void, triggerStart, (), (override));
    MOCK_METHOD(void, triggerStop, (), (override));
    MOCK_METHOD(void, triggerSync, (), (override));
    MOCK_METHOD(void, setNTriggers, (unsigned short n), (override));
    MOCK_METHOD(void, setTrigger,
            (unsigned int timeToNextTrigger, bool syncReq, unsigned short idx, bool syncMode, bool irqDone),
    (override));
    MOCK_METHOD(void, updateFirmware, (const char * filename, bool eraseHVPS), (override));
    MOCK_METHOD(float, getUpdateFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, getUpdateFirmwareStatus, (), (override));
    MOCK_METHOD(int, updateTxFirmware,
            (const char * seaFilename, const char * sedFilename),
    (override));
    MOCK_METHOD(float, getUpdateTxFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, getUpdateTxFirmwareStatus, (), (override));
    MOCK_METHOD(void, swTrigger, (), (override));
    MOCK_METHOD(void, swNextTx, (), (override));
    MOCK_METHOD(void, enableTestPatterns, (), (override));
    MOCK_METHOD(void, disableTestPatterns, (), (override));
    MOCK_METHOD(void, syncTestPatterns, (), (override));
    MOCK_METHOD(void, scheduleTransferRxBufferToHost, (const size_t, unsigned char *, size_t, size_t,
        const std::function<void (void)> &));
    MOCK_METHOD(void, syncTransfer, (), (override));
    MOCK_METHOD(void, scheduleTransferRxBufferToHost, (const size_t,const size_t,const std::function<void (void)> &), (override));
    MOCK_METHOD(void, prepareTransferRxBufferToHost, (const size_t,unsigned char *,size_t,size_t, bool isGpu), (override));
    MOCK_METHOD(void, prepareHostBuffer, (unsigned char *,size_t,size_t, bool isGpu), (override));
    MOCK_METHOD(void, markEntriesAsReadyForReceive, (unsigned short,unsigned short), (override));
    MOCK_METHOD(void, markEntriesAsReadyForTransfer, (unsigned short,unsigned short), (override));
    MOCK_METHOD(void, registerReceiveOverflowCallback, (const std::function<void (void)> &), (override));
    MOCK_METHOD(void, registerTransferOverflowCallback, (const std::function<void (void)> &), (override));
    MOCK_METHOD(void, registerCallback, (IUs4OEM::MSINumber, const std::function<void (void)> &), (override));
    MOCK_METHOD(void, enableWaitOnReceiveOverflow, (), (override));
    MOCK_METHOD(void, enableWaitOnTransferOverflow, (), (override));
    MOCK_METHOD(void, syncReceive, (), (override));
    MOCK_METHOD(void, resetRuntimeCallbacks, (), (override));
    MOCK_METHOD(float, getFpgaTemp, (), (override));
    MOCK_METHOD(void, waitForPendingTransfers, (), (override));
    MOCK_METHOD(void, clearUcdFaults, (), (override));
    MOCK_METHOD(unsigned short, getUcdStatus, (), (override));
    MOCK_METHOD(unsigned char, getUcdStatusByte, (), (override));
    MOCK_METHOD(float, getUcdTemp, (), (override));
    MOCK_METHOD(float, getUcdExtTemp, (), (override));
    MOCK_METHOD(float, getUcdVout, (unsigned char), (override));
    MOCK_METHOD(float, getUcdIout, (unsigned char), (override));
    MOCK_METHOD(unsigned char, getUcdVoutStatus, (unsigned char), (override));
    MOCK_METHOD(unsigned char, getUcdIoutStatus, (unsigned char), (override));
    MOCK_METHOD(unsigned char, getUcdCmlStatus, (unsigned char), (override));
    MOCK_METHOD(std::vector<unsigned char>, getUcdMfrStatus, (unsigned char), (override));
    MOCK_METHOD(std::vector<unsigned char>, getUcdRunTime, (), (override));
    MOCK_METHOD(std::vector<unsigned char>, getUcdBlackBox, (), (override));
    MOCK_METHOD(std::vector<unsigned char>, getUcdLog, (), (override));
    MOCK_METHOD(void, clearUcdLog, (), (override));
    MOCK_METHOD(bool, checkUcdLogNotEmpty, (), (override));
    MOCK_METHOD(void, clearUcdBlackBox, (), (override));
    MOCK_METHOD(void, enableRuntimeInterrupts, (), (override));
    MOCK_METHOD(void, disableRuntimeInterrupts, (), (override));

    MOCK_METHOD(void, afeWriteRegister, (uint8_t, uint8_t, uint16_t), (override));
    MOCK_METHOD(void, afeDemodEnable, (), (override));
    MOCK_METHOD(void, afeDemodEnable, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodDisable, (), (override));
    MOCK_METHOD(void, afeDemodDisable, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDefault, (), (override));
    MOCK_METHOD(void, afeDemodSetDecimationFactor, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDecimationFactorQuarters, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDemodFrequency, (float), (override));
    MOCK_METHOD(void, afeDemodSetDemodFrequency, (float, float), (override));
    MOCK_METHOD(void, afeDemodFsweepEnable, (), (override));
    MOCK_METHOD(void, afeDemodFsweepDisable, (), (override));
    MOCK_METHOD(void, afeDemodSetFsweepRoi, (uint16_t, uint16_t), (override));
    MOCK_METHOD(void, afeDemodSetFirCoeffsBank, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, afeDemodWriteFirCoeffs, (const int16_t*, uint16_t), (override));
    MOCK_METHOD(void, afeDemodWriteFirCoeffs, (const float*, uint16_t), (override));
    MOCK_METHOD(void, afeDemodSetDefault, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDecimationFactor, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDecimationFactorQuarters, (uint8_t, uint8_t, uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetDemodFrequency, (uint8_t, float), (override));
    MOCK_METHOD(void, afeDemodSetDemodFrequency, (uint8_t, float, float), (override));
    MOCK_METHOD(float, afeDemodGetStartFrequency, (), (override));
    MOCK_METHOD(float, afeDemodGetStopFrequency, (), (override));
    MOCK_METHOD(float, afeDemodGetStartFrequency, (uint8_t), (override));
    MOCK_METHOD(float, afeDemodGetStopFrequency, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodFsweepEnable, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodFsweepDisable, (uint8_t), (override));
    MOCK_METHOD(void, afeDemodSetFsweepRoi, (uint8_t, uint16_t, uint16_t), (override));
    MOCK_METHOD(void, afeDemodWriteFirCoeffsBank, (uint8_t, uint32_t*), (override));
    MOCK_METHOD(void, afeDemodWriteFirCoeffs, (uint8_t, const int16_t*, uint16_t), (override));
    MOCK_METHOD(void, afeDemodWriteFirCoeffs, (uint8_t, const float*, uint16_t), (override));
    MOCK_METHOD(uint16_t, afeReadRegister, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, afeSoftReset, (uint8_t), (override));
    MOCK_METHOD(void, afeSoftReset, (), (override));
    MOCK_METHOD(uint32_t, getSequencerConfRegister, (), (override));
    MOCK_METHOD(uint32_t, getSequencerCtrlRegister, (), (override));
    MOCK_METHOD(void, setStandardIoDriveMode, (), (override));
    MOCK_METHOD(void, setWaveformIoDriveMode, (), (override));
    MOCK_METHOD(void, setIoLevels, (uint8_t), (override));
    MOCK_METHOD(void, setFiringIobs, (uint32_t, uint16_t), (override));
    MOCK_METHOD(void, setIobsRegister, (uint16_t, uint8_t, bool, uint16_t), (override));
    MOCK_METHOD(uint32_t, getIobsRegister, (uint16_t), (override));
    MOCK_METHOD(void, listPeriphs, (), (override));
    MOCK_METHOD(void, dumpPeriph, (std::string, uint32_t), (override));
    MOCK_METHOD(size_t, getBaseAddr, (), (override));
    MOCK_METHOD(void, afeSoftTrigger, (), (override));
    MOCK_METHOD(void, afeEnableAutoOffsetRemoval, (), (override));
    MOCK_METHOD(void, afeDisableAutoOffsetRemoval, (), (override));
    MOCK_METHOD(void, afeSetAutoOffsetRemovalCycles, (uint8_t), (override));
    MOCK_METHOD(void, afeSetAutoOffsetRemovalDelay, (uint8_t), (override));
    MOCK_METHOD(uint64_t, getFpgaWallclock, (), (override));
    MOCK_METHOD(void, afeEnableLnaHpf, (), (override));
    MOCK_METHOD(void, afeDisableLnaHpf, (), (override));
    MOCK_METHOD(void, afeSetLnaHpfCornerFrequency, (uint32_t), (override));
    MOCK_METHOD(void, afeEnableAdcHpf, (), (override));
    MOCK_METHOD(void, afeDisableAdcHpf, (), (override));
    MOCK_METHOD(void, afeSetAdcHpfCornerFrequency, (uint32_t), (override));
    MOCK_METHOD(void, afeSetAdcHpfParams, (uint16_t, uint16_t, uint16_t, uint16_t), (override));
    MOCK_METHOD(void, afeDemodConfig, (uint8_t, uint8_t, const float*, uint16_t, float, bool), (override));
    MOCK_METHOD(void, afeDemodConfig, (uint8_t, uint8_t, uint8_t, const float*, uint16_t, float, bool), (override));
    MOCK_METHOD(void, disableWaitOnReceiveOverflow, (), (override));
    MOCK_METHOD(void, disableWaitOnTransferOverflow, (), (override));
    MOCK_METHOD(void, verifyFirmware, (const char*), (override));
    MOCK_METHOD(void, setTxFrequencyRange, (int range), (override));
    MOCK_METHOD(float, getMinTxFrequency, (), (const, override));
    MOCK_METHOD(float, getMaxTxFrequency, (), (const, override));
    MOCK_METHOD(void, pulserWriteRegister, (uint8_t, uint16_t, uint16_t), (override));
    MOCK_METHOD(uint16_t, pulserReadRegister, (uint8_t, uint16_t), (override));
    MOCK_METHOD(uint32_t, getOemVersion, (), (override));
    MOCK_METHOD(void, sequencerWriteRegister, (uint32_t, uint32_t), (override));
    MOCK_METHOD(uint32_t, sequencerReadRegister, (uint32_t), (override));
    MOCK_METHOD(void, allPulsersWriteRegister, (uint16_t, uint16_t), (override));
    MOCK_METHOD(void, dbarLitePcieWriteReg, (uint8_t, uint8_t), (override));
    MOCK_METHOD(uint8_t, dbarLitePcieReadReg, (uint8_t), (override));
    MOCK_METHOD(void, dbarLitePcieWriteBuf, (uint8_t, std::vector<unsigned char>), (override));
    MOCK_METHOD(void, dbarLitePcieReadBuf, (uint8_t, std::vector<unsigned char>), (override));
    MOCK_METHOD(void, hvpsWriteRegister, (uint32_t, uint32_t), (override));
    MOCK_METHOD(uint32_t, hvpsReadRegister, (uint32_t), (override));
    MOCK_METHOD(void, hvpsSetVoltage, (float), (override));
    MOCK_METHOD(IHV*, getHvps, (), (override));
    MOCK_METHOD(std::string, getSerialNumber, (), (override));
    MOCK_METHOD(std::string, getRevisionNumber, (), (override));
    MOCK_METHOD(void, enableProbeCheck, (uint8_t), (override));
    MOCK_METHOD(bool, checkProbeConnected, (), (override));
    MOCK_METHOD(void, disableProbeCheck, (), (override));
    MOCK_METHOD(float, getMinTxPulseLength, (), (const, override));
    MOCK_METHOD(float, getMaxTxPulseLength, (), (const, override));
    MOCK_METHOD(void, setSubsequences, (const std::vector<uint16_t> &start, const std::vector<uint16_t> &end, bool syncMode, const std::vector<uint32_t> &endTimeToNextTrigger), (override));
    MOCK_METHOD(void, resetSequencer, (), (override));
    MOCK_METHOD(float, setHvpsSyncMeasurement, (uint16_t, float), (override));
    MOCK_METHOD(HVPSMeasurements, getHvpsMeasurements, (), (override));
    MOCK_METHOD(void, enableHvpsMeasurementReadyIrq, (), (override));
    MOCK_METHOD(void, disableHvpsMeasurementReadyIrq, (), (override));
    MOCK_METHOD(void, clearTransferRxBufferToHost, (const size_t firing), (override));
    MOCK_METHOD(void, verifyTxWaveform, (), (override));
    MOCK_METHOD(void, enableTxTimeout, (), (override));
    MOCK_METHOD(void, disableTxTimeout, (), (override));
    MOCK_METHOD(void, setTxTimeout, (uint8_t id, uint16_t timeoutUs), (override));
    MOCK_METHOD(void, setFiringTxTimoutId, (uint16_t firing, uint8_t id), (override));
    MOCK_METHOD(void, setTxVoltageLevel, (uint8_t level, uint16_t firing), (override));
    MOCK_METHOD(float, getOcwsFrequency, (const float frequency), (override));
    MOCK_METHOD(void, logPulsersInterruptRegister, (), (override));
    MOCK_METHOD(void, setCustomSequenceWaveform, (const unsigned short firing, const std::vector<uint32_t>&), (override));
    MOCK_METHOD(float, getMeasuredHvmVoltage, (), (override));
    MOCK_METHOD(float, getMeasuredHvpVoltage, (), (override));
    MOCK_METHOD(float, getMeasuredHvp0Voltage, (), (override));
    MOCK_METHOD(float, getMeasuredHvp1Voltage, (), (override));
    MOCK_METHOD(float, getMeasuredHvm0Voltage, (), (override));
    MOCK_METHOD(float, getMeasuredHvm1Voltage, (), (override));
    MOCK_METHOD(std::vector<float>, getMeasuredVoltages, (), (override));
    MOCK_METHOD((std::pair<float, float>), getTgcValueRange, (), (const, override));
    MOCK_METHOD(void, buildSequenceWaveforms, (bool verify), (override));
    MOCK_METHOD(::us4us::us4r::Vector<uint32_t>, runPulserReadbackTest, (uint32_t), (override));
    MOCK_METHOD(void, disableWatchdog, (), (override));
    MOCK_METHOD(void, setOemWatchdogThresholds, (uint16_t threshold0, uint16_t threshold1), (override));
    MOCK_METHOD(void, setHostWatchdogThresholds, (uint16_t threshold), (override));
    MOCK_METHOD(void, enableInterrupts, (), (override));
    MOCK_METHOD(void, enableSystemInterrupts, (const IUs4OEM::CallbacksMap&), (override));
    MOCK_METHOD(void, resetDmaCallbacks, (), (override));
    MOCK_METHOD(void, setPulserInterruptCallback, (const std::function<void()> &), (override));
    MOCK_METHOD(void, eraseHvpsTuningVector, (), (override));
    MOCK_METHOD(void, tuneHvps, (uint8_t limit), (override));
    MOCK_METHOD(bool, isHvpsTuned, (), (override));
    MOCK_METHOD(void, buildSequenceWaveform, (const unsigned short), (override));
    MOCK_METHOD(uint16_t, getSequencerCurrentIndex, (), (override));
    MOCK_METHOD(bool, isEntryReadyForTransfer, (uint16_t), (override));
    MOCK_METHOD(bool, isEntryReadyForReceive, (uint16_t), (override));
    MOCK_METHOD(void, waitForSequencerIdle, (), (override));
    MOCK_METHOD(std::vector<uint16_t>, getPulsersStatusRegister, (), (override));
    MOCK_METHOD(std::vector<std::string>, getPulserStatusRegisterDescription, (uint16_t status), (override));
    MOCK_METHOD(void, setHvpsVoltage, (uint8_t), (override));
    MOCK_METHOD(void, setCustomHvpsFuseThresholds, (::us4us::us4r::HvpsRails rail, const ::us4us::us4r::HvpsFuseCustomThresholds& thresholds), (override));
    MOCK_METHOD(int64_t, getHvpsTuningTimestamp, (), (override));
    MOCK_METHOD(void, setIoDirection, (uint8_t io, IUs4OEM::Direction direction), (override));
};

#define GET_MOCK_PTR(sptr) *(MockIUs4OEM *) (sptr.get())

#endif //ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
