#pragma once

#include <boost/any.hpp>
#include "Utils/DispatcherLogger.h"

class VariableAnyValue {
private:
    boost::any value;
    bool isPointer;
public:
    VariableAnyValue() {}

    VariableAnyValue(const boost::any &value, const bool isPointer = false) {
        this->value = value;
        this->isPointer = isPointer;
    }

    ~VariableAnyValue() {}

    template<typename T>
    T getValue() const {
        try {
            if(this->isPointer)
                return boost::any_cast<T>(*(boost::any_cast<boost::any *>(this->value)));
            return boost::any_cast<T>(this->value);
        }
        catch(boost::bad_any_cast exception) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Bad variable type cast with generic value."));
            return static_cast<T>(0);
        }
    }

    template<typename T>
    T *getValuePtr() {
        try {
            if(this->isPointer)
                return boost::any_cast<T>(boost::any_cast<boost::any *>(this->value));
            return boost::any_cast<T>(&this->value);
        }
        catch(boost::bad_any_cast exception) {
            DISPATCHER_LOG(DispatcherLogType::ERROR_, std::string("Bad variable type cast with generic value."));
            return nullptr;
        }
    }

    boost::any *getAnyValuePtr() {
        return &(this->value);
    }

    boost::any getAnyValue() const {
        return this->value;
    }

    boost::any getExpandedAnyValue() {
        if(this->isPointer)
            return *(boost::any_cast<boost::any *>(this->value));
        else
            return this->value;
    }

    void setValue(const boost::any &value) {
        this->value = value;
    }
};

