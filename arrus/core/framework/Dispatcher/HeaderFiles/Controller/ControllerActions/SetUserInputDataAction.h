#pragma once

#include "Controller/ControllerActions/ControllerAction.h"
#include "Controller/ControllerEvents/ControllerEvent.h"
#include "Controller/ControllerEvents/SetUserInputDataEvent.h"
#include "Controller/Controller.h"
#include "Utils/DispatcherLogger.h"

template<typename T>
class SetUserInputDataAction : public ControllerAction {
private:
    DataPtr prepareUserInputData(T *userDataPtr, const int numberOfBatches, const Dims dims) {
        unsigned int dataCount = dims.flatten() * numberOfBatches;
        T *data = new T[dataCount];
        memcpy(data, userDataPtr, dataCount * sizeof(T));
        DataPtr dataPtr(data, dims);
        dataPtr.setPtrProperty(std::string("numberOfBatches"), VariableAnyValue(numberOfBatches));
        if(typeid(T) == typeid(float2)) {
            dataPtr.setPtrProperty(std::string("iq"), VariableAnyValue(true));
        } else {
            dataPtr.setPtrProperty(std::string("iq"), VariableAnyValue(false));
        }
        return dataPtr;
    }

public:
    SetUserInputDataAction(Controller *controller) : ControllerAction(controller) {};

    ~SetUserInputDataAction() {};

    void performAction(std::shared_ptr <IControllerEvent> controllerEvent) {
        std::shared_ptr <SetUserInputDataEvent<T>> setUserInputDataEvent =
            std::dynamic_pointer_cast < SetUserInputDataEvent < T >> (controllerEvent);

        DataPtr userData = this->prepareUserInputData(setUserInputDataEvent->getDataPtr(),
                                                      setUserInputDataEvent->getNumberOfBatches(),
                                                      setUserInputDataEvent->getDims());
        this->controller->getModel()->activateUserDataProvider(userData);
    }
};

