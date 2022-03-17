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
            (unsigned int timeToNextTrigger, bool syncReq, unsigned short idx),
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
    MOCK_METHOD(void, SetTxFrequencyRange, (int), (override));
    MOCK_METHOD(int, GetTxFrequencyRange, (), (override));
};

#define GET_MOCK_PTR(sptr) *(MockIUs4OEM *) (sptr.get())

#endif //ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
