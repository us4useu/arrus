#ifndef ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
#define ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <ius4oem.h>

class MockIUs4OEM : public IUs4OEM {
public:
    MOCK_METHOD(unsigned int, GetID, (), (override));
    MOCK_METHOD(bool, IsPowereddown, (), (override));
    MOCK_METHOD(void, Initialize, (int), (override));
    MOCK_METHOD(void, Synchronize, (), (override));
    MOCK_METHOD(void, ScheduleReceive,
            (const size_t firing, const size_t address, const size_t length, const uint32_t start, const uint32_t decimation, const size_t rxMapId, const std::function<void()>& callback),
    (override));
    MOCK_METHOD(void, ClearScheduledReceive, (), (override));
    MOCK_METHOD(void, TransferRXBufferToHost,
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
    MOCK_METHOD(void, EnableSequencer, (), (override));
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
            (unsigned short timeToNextTrigger, bool syncReq, unsigned short idx),
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
    MOCK_METHOD(void, LockDMABuffer, (unsigned char * address, size_t length),
    (override));
    MOCK_METHOD(void, ReleaseDMABuffer, (unsigned char * address), (override));
};

#define GET_MOCK_PTR(sptr) *(MockIUs4OEM *) (sptr.get())

#endif //ARRUS_CORE_DEVICES_US4R_TESTS_MOCKIUS4OEM_H
