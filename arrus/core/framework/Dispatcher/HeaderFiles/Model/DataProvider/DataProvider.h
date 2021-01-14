#pragma once

#include "Model/IntelligentBuffer.h"
#include <boost/thread.hpp>
#include <unordered_map>
#include <vector>
#include "Model/VariableAnyValue.h"
#include "jsonConfigParser.h"

typedef std::unordered_map <std::string, VariableAnyValue> anyValueMap;

class DataProvider {
private:
    void process();

    void readGlobalTransmitInfo(const JsonConfigData &jsonData);

    void readFramesInfo(const JsonConfigData &jsonData);

    void readTransmitFrequencies(const std::vector <Event> &events, const int frameId,
                                 const std::vector <TransmitWaveform> &waveforms);

    void readEventsInfo(const std::vector <Event> &events, const int frameId,
                        const std::vector <TransmitWaveform> &waveforms);

    void readSteeringAngles(const std::vector <Event> &events, const int frameId);

    void readOriginTransmitters(const std::vector <Event> &events, const int frameId);

    void readTransmitApertures(const std::vector <Event> &events, const int frameId);

    void readFocuses(const std::vector <Event> &events, const int frameId);

    void readSamplesRange(const Frame &frame, const int frameId);

protected:
    IntelligentBuffer *intelligentBuffer;
    bool isWorking;
    boost::thread myThread;
    int dataIdCounter;

    anyValueMap globalTransmitInfo;
    std::vector <anyValueMap> framesInfo;

    virtual void customProcess() = 0;

    virtual void preStart(const bool startOnce) {};

    virtual void preStop() {};

    virtual void preKill() {};

    void readTransmitInfo(const std::string &jsonConfig);

    void addTransmitInfoToDataPtr(DataPtr &dataPtr, const int frameId);

public:
    DataProvider();

    virtual ~DataProvider() {};

    void start(const bool startOnce);

    void stop();

    void kill();

    virtual int getBatchCount() = 0;
};

