#pragma once

#include "Controller/ControllerEvents/ControllerEvent.h"

template<typename T>
class SetUserInputDataEvent : public ControllerEvent<SetUserInputDataEvent<T>> {
private:
    Dims dims;
    int numberOfBatches;
    T *dataPtr;

public:
    SetUserInputDataEvent(T *dataPtr, const int numberOfBatches, const Dims dims) {
        this->dataPtr = dataPtr;
        this->numberOfBatches = numberOfBatches;
        this->dims = dims;
    }

    ~SetUserInputDataEvent() {};

    Dims getDims() {
        return this->dims;
    }

    int getNumberOfBatches() {
        return this->numberOfBatches;
    }

    T *getDataPtr() {
        return this->dataPtr;
    }
};

