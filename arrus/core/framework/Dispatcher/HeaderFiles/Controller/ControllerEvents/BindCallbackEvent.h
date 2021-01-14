#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

#include <boost/function.hpp>

class BindCallbackEvent : public ControllerEvent<BindCallbackEvent> {
private:
    const int graphNodeId;
    const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                               int dataType)> callbackFunction;

public:
    BindCallbackEvent(const int id,
                      const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY,
                                                 int dimZ, int dataType)> callback);

    ~BindCallbackEvent();

    const int getNodeId();

    const boost::function<void(void *data, int iterationId, int graphNodeId, int dimX, int dimY, int dimZ,
                               int dataType)> getCallback();
};

