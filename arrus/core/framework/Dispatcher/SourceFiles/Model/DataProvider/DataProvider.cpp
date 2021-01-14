#include "Model/DataProvider/DataProvider.h"
#include <sstream>
#include <boost/foreach.hpp>
#include "Utils/DispatcherLogger.h"


#define _USE_MATH_DEFINES

#include <cmath>

DataProvider::DataProvider() {
    this->isWorking = false;
    this->dataIdCounter = 0;
}

void DataProvider::process() {
    while(this->isWorking) {
        this->customProcess();
    }
}

void DataProvider::start(const bool startOnce) {
    this->preStart(startOnce);
    if(!this->isWorking) {
        this->isWorking = true;
        this->myThread = boost::thread(&DataProvider::process, this);
    }
}

void DataProvider::stop() {
    this->isWorking = false;
    this->preStop();
    this->intelligentBuffer->unlockWriterToExit();
    this->myThread.join();
}

void DataProvider::kill() {
    this->isWorking = false;
    this->preKill();
    this->myThread.interrupt();
}

void DataProvider::readGlobalTransmitInfo(const JsonConfigData &jsonData) {
    this->globalTransmitInfo["speedOfSound"] = VariableAnyValue(jsonData.sos);
    this->globalTransmitInfo["samplingFrequency"] = VariableAnyValue(static_cast<float>(jsonData.samplingFrequency));
    this->globalTransmitInfo["numReceivers"] = VariableAnyValue(jsonData.transducer.numElements);
    this->globalTransmitInfo["receiverWidth"] = VariableAnyValue(
        jsonData.transducer.pitch * (jsonData.transducer.numElements - 1));
    this->globalTransmitInfo["pitch"] = VariableAnyValue(jsonData.transducer.pitch);
}

void DataProvider::readTransmitInfo(const std::string &jsonConfig) {
    JsonConfigParser jsp;
    JsonConfigData configData = jsp.parseJson(jsonConfig);

    this->readGlobalTransmitInfo(configData);
    this->readFramesInfo(configData);
}

void DataProvider::readSteeringAngles(const std::vector <Event> &events, const int frameId) {
    std::vector<float> steeringAngles(events.size());
    for(int i = 0; i < events.size(); ++i) {
        if(events[i].coordType.compare("polar") == 0)
            steeringAngles[i] = events[i].focus.theta;
        else if(events[i].coordType.compare("cartesian") == 0) {
            steeringAngles[i] = atan2f(events[i].focus.x - events[i].center.x,
                                       events[i].focus.y - events[i].center.y) * 180.0f / (float) M_PI;
        } else /* file */
            DISPATCHER_LOG(DispatcherLogType::WARNING,
                           std::string("The steering angles are not set when using delays from file."));
    }

    this->framesInfo[frameId]["steeringAngles"] = VariableAnyValue(steeringAngles);
}

void DataProvider::readOriginTransmitters(const std::vector <Event> &events, const int frameId) {
    std::vector<int> originTransmitters(events.size());
    for(int i = 0; i < events.size(); ++i) {
        originTransmitters[i] = events[i].origin;
    }

    this->framesInfo[frameId]["originTransmitters"] = VariableAnyValue(originTransmitters);
}

void DataProvider::readTransmitApertures(const std::vector <Event> &events, const int frameId) {
    std::vector<int> transmitApertures(events.size());
    for(int i = 0; i < events.size(); ++i) {
        transmitApertures[i] = events[i].aperture;
    }

    this->framesInfo[frameId]["transmitApertures"] = VariableAnyValue(transmitApertures);
}

void DataProvider::readFocuses(const std::vector <Event> &events, const int frameId) {
    std::vector <float2> focuses(events.size());
    for(int i = 0; i < events.size(); ++i) {
        if(events[i].coordType.compare("polar") == 0) {
            float r = events[i].focus.r;
            float theta = events[i].focus.theta / 180.0f * (float) M_PI;
            focuses[i] = make_float2(r * sinf(theta), r * cosf(theta));
        } else if(events[i].coordType.compare("cartesian") == 0)
            focuses[i] = make_float2(events[i].focus.x, events[i].focus.y);
        else /* file */
            DISPATCHER_LOG(DispatcherLogType::WARNING,
                           std::string("The focusing points are not set when using delays from file."));
    }

    this->framesInfo[frameId]["focuses"] = VariableAnyValue(focuses);
}

void DataProvider::readTransmitFrequencies(const std::vector <Event> &events, const int frameId,
                                           const std::vector <TransmitWaveform> &waveforms) {
    std::unordered_map<int, float> transmitFrequencies;
    for(TransmitWaveform tw : waveforms) {
        int id = 0; // in future each waveform will have his own id
        transmitFrequencies[id] = static_cast<float>(tw.clock) / (tw.a + tw.b + tw.c + tw.b + 4);
    }

    std::vector<float> transmitFrequenciesForEvents(events.size());
    for(int i = 0; i < events.size(); ++i) {
        int waveformIdx = 0; // in future each event might have his own waveform id
        transmitFrequenciesForEvents[i] = transmitFrequencies[waveformIdx];
    }

    this->framesInfo[frameId]["transmitFrequencies"] = VariableAnyValue(transmitFrequenciesForEvents);
}

void DataProvider::readEventsInfo(const std::vector <Event> &events, const int frameId,
                                  const std::vector <TransmitWaveform> &waveforms) {
    this->framesInfo[frameId]["eventsCount"] = VariableAnyValue(static_cast<int>(events.size()));

    this->readTransmitFrequencies(events, frameId, waveforms);
    this->readSteeringAngles(events, frameId);
    this->readOriginTransmitters(events, frameId);
    this->readTransmitApertures(events, frameId);
    this->readFocuses(events, frameId);
}

void DataProvider::readSamplesRange(const Frame &frame, const int frameId) {
    float startDepth = frame.startSample * this->globalTransmitInfo["speedOfSound"].getValue<float>() /
                       this->globalTransmitInfo["samplingFrequency"].getValue<float>() * 0.5f;

    this->framesInfo[frameId]["samplesCount"] = VariableAnyValue(frame.endSample - frame.startSample);
    this->framesInfo[frameId]["startDepth"] = VariableAnyValue(startDepth);
}

void DataProvider::readFramesInfo(const JsonConfigData &jsonData) {
    this->framesInfo.resize(jsonData.frames.size());

    for(std::pair<int, Frame> f : jsonData.frames) {
        this->readSamplesRange(f.second, f.first);
        this->readEventsInfo(f.second.receiveEvents, f.first, jsonData.transmitWaveforms);
    }
}

void DataProvider::addTransmitInfoToDataPtr(DataPtr &dataPtr, const int frameId) {
    dataPtr.setPtrProperties(this->globalTransmitInfo);
    dataPtr.setPtrProperties(this->framesInfo[frameId]);
    dataPtr.setPtrProperty("frameId", VariableAnyValue(frameId));
    dataPtr.setPtrProperty("dataId", VariableAnyValue(this->dataIdCounter++));
    dataPtr.setPtrProperty("iq", VariableAnyValue(false));
    dataPtr.setPtrProperty("rawDataFromHAL", VariableAnyValue(true));
}
