#ifndef ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
#define ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <ius4oem.h>

class MockIUs4OEM : public IUs4OEM {
public:
    MOCK_METHOD(unsigned int, GetID, (), (override));
    MOCK_METHOD(uint32_t, GetFirmwareVersion, (), (override));
    MOCK_METHOD(uint32_t, GetTxFirmwareVersion, (), (override));
    MOCK_METHOD(void, CheckFirmwareVersion, (), (override));
    MOCK_METHOD(bool, IsPowereddown, (), (override));
    MOCK_METHOD(void, Initialize, (int), (override));
    MOCK_METHOD(void, Synchronize, (), (override));
    MOCK_METHOD(void, ScheduleReceive,
            (const size_t firing, const size_t address, const size_t length, const uint32_t start, const uint32_t decimation, const size_t rxMapId, const std::function<void()>& callback),
    (override));
    MOCK_METHOD(void, ClearScheduledReceive, (), (override));
    MOCK_METHOD(void, TransferRXBufferToHost,
            (unsigned char * dstAddress, size_t length, size_t srcAddress, bool isGpu),
    (override));
    MOCK_METHOD(void, ReleaseTransferRxBufferToHost,
        (unsigned char * dstAddress, size_t length, size_t srcAddress),
    (override));
    MOCK_METHOD(void, SetPGAGain, (us4r::afe58jd18::PGA_GAIN gain), (override));
    MOCK_METHOD(void, SetLPFCutoff, (us4r::afe58jd18::LPF_PROG cutoff),
    (override));
    MOCK_METHOD(void, SetActiveTermination,
            (us4r::afe58jd18::ACTIVE_TERM_EN endis, us4r::afe58jd18::GBL_ACTIVE_TERM term),
    (override));
    MOCK_METHOD(void, SetLNAGain, (us4r::afe58jd18::LNA_GAIN_GBL gain),
    (override));
    MOCK_METHOD(void, SetDTGC,
            (us4r::afe58jd18::EN_DIG_TGC endis, us4r::afe58jd18::DIG_TGC_ATTENUATION att),
    (override));
    MOCK_METHOD(void, InitializeTX, (), (override));
    MOCK_METHOD(void, SetNumberOfFirings, (const unsigned short nFirings),
    (override));
    MOCK_METHOD(float, SetTxDelay,
            (const unsigned char channel, const float value, const unsigned short firing),
    (override));
    MOCK_METHOD(float, SetTxFreqency,
            (const float frequency, const unsigned short firing),
    (override));
    MOCK_METHOD(unsigned char, SetTxHalfPeriods,
            (unsigned char nop, const unsigned short firing), (override));
    MOCK_METHOD(void, SetTxInvert, (bool onoff, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetTxCw, (bool onoff, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetRxAperture,
            (const unsigned char origin, const unsigned char size, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetTxAperture,
            (const unsigned char origin, const unsigned char size, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetRxAperture,
            (const std::bitset<NCH>& aperture, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetTxAperture,
            (const std::bitset<NCH>& aperture, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetActiveChannelGroup,
            (const std::bitset<NCH / 8>& group, const unsigned short firing),
    (override));
    MOCK_METHOD(void, SetRxTime,
            (const float time, const unsigned short firing), (override));
    MOCK_METHOD(void, SetRxDelay,
            (const float delay, const unsigned short firing), (override));
    MOCK_METHOD(void, EnableTransmit, (), (override));
    MOCK_METHOD(void, EnableSequencer, (bool txConfOnTrigger), (override));
    MOCK_METHOD(void, SetRxChannelMapping,
            ( const std::vector<uint8_t> & mapping, const uint16_t rxMapId),
    (override));
    MOCK_METHOD(void, SetTxChannelMapping,
            (const unsigned char srcChannel, const unsigned char dstChannel),
    (override));
    MOCK_METHOD(void, TGCEnable, (), (override));
    MOCK_METHOD(void, TGCDisable, (), (override));
    MOCK_METHOD(void, TGCSetSamples,
            (const std::vector<float> & samples, const int firing),
    (override));
    MOCK_METHOD(void, TriggerStart, (), (override));
    MOCK_METHOD(void, TriggerStop, (), (override));
    MOCK_METHOD(void, TriggerSync, (), (override));
    MOCK_METHOD(void, SetNTriggers, (unsigned short n), (override));
    MOCK_METHOD(void, SetTrigger,
            (unsigned int timeToNextTrigger, bool syncReq, unsigned short idx, bool syncMode),
    (override));
    MOCK_METHOD(void, UpdateFirmware, (const char * filename), (override));
    MOCK_METHOD(float, GetUpdateFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, GetUpdateFirmwareStatus, (), (override));
    MOCK_METHOD(int, UpdateTxFirmware,
            (const char * seaFilename, const char * sedFilename),
    (override));
    MOCK_METHOD(float, GetUpdateTxFirmwareProgress, (), (override));
    MOCK_METHOD(const char *, GetUpdateTxFirmwareStatus, (), (override));
    MOCK_METHOD(void, SWTrigger, (), (override));
    MOCK_METHOD(void, SWNextTX, (), (override));
    MOCK_METHOD(void, EnableTestPatterns, (), (override));
    MOCK_METHOD(void, DisableTestPatterns, (), (override));
    MOCK_METHOD(void, SyncTestPatterns, (), (override));
    MOCK_METHOD(void, LockDMABuffer, (unsigned char * address, size_t length, bool isGpu),
    (override));
    MOCK_METHOD(void, ReleaseDMABuffer, (unsigned char * address), (override));
    MOCK_METHOD(void, ScheduleTransferRXBufferToHost, (const size_t, unsigned char *, size_t, size_t,
        const std::function<void (void)> &));
    MOCK_METHOD(void, SyncTransfer, (), (override));
    MOCK_METHOD(void, ScheduleTransferRXBufferToHost, (const size_t,const size_t,const std::function<void (void)> &), (override));
    MOCK_METHOD(void, PrepareTransferRXBufferToHost, (const size_t,unsigned char *,size_t,size_t, bool isGpu), (override));
    MOCK_METHOD(void, PrepareHostBuffer, (unsigned char *,size_t,size_t, bool isGpu), (override));
    MOCK_METHOD(void, MarkEntriesAsReadyForReceive, (unsigned short,unsigned short), (override));
    MOCK_METHOD(void, MarkEntriesAsReadyForTransfer, (unsigned short,unsigned short), (override));
    MOCK_METHOD(void, RegisterReceiveOverflowCallback, (const std::function<void (void)> &), (override));
    MOCK_METHOD(void, RegisterTransferOverflowCallback, (const std::function<void (void)> &), (override));
    MOCK_METHOD(void, EnableWaitOnReceiveOverflow, (), (override));
    MOCK_METHOD(void, EnableWaitOnTransferOverflow, (), (override));
    MOCK_METHOD(void, SyncReceive, (), (override));
    MOCK_METHOD(void, ResetCallbacks, (), (override));
    MOCK_METHOD(float, GetFPGATemp, (), (override));
    MOCK_METHOD(void, WaitForPendingTransfers, (), (override));
    MOCK_METHOD(void, ClearUCDFaults, (), (override));
    MOCK_METHOD(unsigned short, GetUCDStatus, (), (override));
    MOCK_METHOD(unsigned char, GetUCDStatusByte, (), (override));
    MOCK_METHOD(float, GetUCDTemp, (), (override));
    MOCK_METHOD(float, GetUCDExtTemp, (), (override));
    MOCK_METHOD(float, GetUCDVOUT, (unsigned char), (override));
    MOCK_METHOD(float, GetUCDIOUT, (unsigned char), (override));
    MOCK_METHOD(unsigned char, GetUCDVOUTStatus, (unsigned char), (override));
    MOCK_METHOD(unsigned char, GetUCDIOUTStatus, (unsigned char), (override));
    MOCK_METHOD(unsigned char, GetUCDCMLStatus, (unsigned char), (override));
    MOCK_METHOD(std::vector<unsigned char>, GetUCDMFRStatus, (unsigned char), (override));
    MOCK_METHOD(std::vector<unsigned char>, GetUCDRunTime, (), (override));
    MOCK_METHOD(std::vector<unsigned char>, GetUCDBlackBox, (), (override));
    MOCK_METHOD(std::vector<unsigned char>, GetUCDLog, (), (override));
    MOCK_METHOD(void, ClearUCDLog, (), (override));
    MOCK_METHOD(bool, CheckUCDLogNotEmpty, (), (override));
    MOCK_METHOD(void, ClearUCDBlackBox, (), (override));
    MOCK_METHOD(void, EnableInterrupts, (), (override));
    MOCK_METHOD(void, DisableInterrupts, (), (override));

    MOCK_METHOD(void, SetActiveTermination, (us4r::afe58jd18::ACTIVE_TERM_EN, us4r::afe58jd18::ACT_TERM_IND_RES), (override));
    MOCK_METHOD(void, AfeWriteRegister, (uint8_t, uint8_t, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodEnable, (), (override));
    MOCK_METHOD(void, AfeDemodEnable, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodDisable, (), (override));
    MOCK_METHOD(void, AfeDemodDisable, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDefault, (), (override));
    MOCK_METHOD(void, AfeDemodSetDecimationFactor, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDecimationFactorQuarters, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDemodFrequency, (float), (override));
    MOCK_METHOD(void, AfeDemodSetDemodFrequency, (float, float), (override));
    MOCK_METHOD(void, AfeDemodFsweepEnable, (), (override));
    MOCK_METHOD(void, AfeDemodFsweepDisable, (), (override));
    MOCK_METHOD(void, AfeDemodSetFsweepROI, (uint16_t, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodSetFirCoeffsBank, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, AfeDemodWriteFirCoeffs, (const int16_t*, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodWriteFirCoeffs, (const float*, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodSetDefault, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDecimationFactor, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDecimationFactorQuarters, (uint8_t, uint8_t, uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetDemodFrequency, (uint8_t, float), (override));
    MOCK_METHOD(void, AfeDemodSetDemodFrequency, (uint8_t, float, float), (override));
    MOCK_METHOD(float, AfeDemodGetStartFrequency, (), (override));
    MOCK_METHOD(float, AfeDemodGetStopFrequency, (), (override));
    MOCK_METHOD(float, AfeDemodGetStartFrequency, (uint8_t), (override));
    MOCK_METHOD(float, AfeDemodGetStopFrequency, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodFsweepEnable, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodFsweepDisable, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodSetFsweepROI, (uint8_t, uint16_t, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodWriteFirCoeffsBank, (uint8_t, uint32_t*), (override));
    MOCK_METHOD(void, AfeDemodWriteFirCoeffs, (uint8_t, const int16_t*, uint16_t), (override));
    MOCK_METHOD(void, AfeDemodWriteFirCoeffs, (uint8_t, const float*, uint16_t), (override));
    MOCK_METHOD(uint16_t, AfeReadRegister, (uint8_t, uint8_t), (override));
    MOCK_METHOD(void, AfeSoftReset, (uint8_t), (override));
    MOCK_METHOD(void, AfeSoftReset, (), (override));
    MOCK_METHOD(void, WaitForPendingInterrupts, (), (override));
    MOCK_METHOD(uint32_t, GetSequencerConfRegister, (), (override));
    MOCK_METHOD(uint32_t, GetSequencerCtrlRegister, (), (override));
    MOCK_METHOD(void, SetStandardIODriveMode, (), (override));
    MOCK_METHOD(void, SetWaveformIODriveMode, (), (override));
    MOCK_METHOD(void, SetIOLevels, (uint8_t), (override));
    MOCK_METHOD(void, SetFiringIOBS, (uint32_t, uint8_t), (override));
    MOCK_METHOD(void, SetIOBSRegister, (uint8_t, uint8_t, uint8_t, bool, uint16_t), (override));
    MOCK_METHOD(void, ListPeriphs, (), (override));
    MOCK_METHOD(void, DumpPeriph, (std::string, uint32_t), (override));
    MOCK_METHOD(size_t, GetBaseAddr, (), (override));
    MOCK_METHOD(void, AfeSoftTrigger, (), (override));
    MOCK_METHOD(void, AfeEnableAutoOffsetRemoval, (), (override));
    MOCK_METHOD(void, AfeDisableAutoOffsetRemoval, (), (override));
    MOCK_METHOD(void, AfeSetAutoOffsetRemovalCycles, (uint8_t), (override));
    MOCK_METHOD(void, AfeSetAutoOffsetRemovalDelay, (uint8_t), (override));
    MOCK_METHOD(float, GetFPGAWallclock, (), (override));
    MOCK_METHOD(void, AfeEnableHPF, (), (override));
    MOCK_METHOD(void, AfeDisableHPF, (), (override));
    MOCK_METHOD(void, AfeSetHPFCornerFrequency, (uint8_t), (override));
    MOCK_METHOD(void, AfeDemodConfig, (uint8_t, uint8_t, const float*, uint16_t, float), (override));
    MOCK_METHOD(void, AfeDemodConfig, (uint8_t, uint8_t, uint8_t, const float*, uint16_t, float), (override));
    MOCK_METHOD(void, DisableWaitOnReceiveOverflow, (), (override));
    MOCK_METHOD(void, DisableWaitOnTransferOverflow, (), (override));
    MOCK_METHOD(void, VerifyFirmware, (const char*), (override));
    MOCK_METHOD(void, SetTxFrequencyRange, (int range), (override));
    MOCK_METHOD(float, GetMinTxFrequency, (), (const, override));
    MOCK_METHOD(float, GetMaxTxFrequency, (), (const, override));
    MOCK_METHOD(void, PulserWriteRegister, (uint8_t, uint16_t, uint16_t), (override));
    MOCK_METHOD(uint16_t, PulserReadRegister, (uint8_t, uint16_t), (override));
    MOCK_METHOD(void, SequencerWriteRegister, (uint32_t, uint32_t), (override));
    MOCK_METHOD(uint32_t, SequencerReadRegister, (uint32_t), (override));
    MOCK_METHOD(void, AllPulsersWriteRegister, (uint16_t, uint16_t), (override));
    MOCK_METHOD(uint32_t, GetTxOffset, (), (override));
};

#define GET_MOCK_PTR(sptr) *(MockIUs4OEM *) (sptr.get())

#endif //ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
