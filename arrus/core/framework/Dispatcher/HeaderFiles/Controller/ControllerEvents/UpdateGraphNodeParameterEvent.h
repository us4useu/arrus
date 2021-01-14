#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"
#include <string>
#include "Model/VariableAnyValue.h"

class UpdateGraphNodeParameterEvent : public ControllerEvent<UpdateGraphNodeParameterEvent> {
private:
    int nodeId;
    std::string parameterName;
    VariableAnyValue parameterValue;
public:
    UpdateGraphNodeParameterEvent(const int nodeId, const std::string &parameterName,
                                  const VariableAnyValue parameterValue);

    ~UpdateGraphNodeParameterEvent();

    int getNodeId();

    const std::string &getParameterName();

    const VariableAnyValue &getParameterValue();
};

