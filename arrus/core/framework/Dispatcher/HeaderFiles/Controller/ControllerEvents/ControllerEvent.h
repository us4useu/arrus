#pragma once

// Curiously recurring template pattern //

static unsigned int nextID = 0;

class IControllerEvent {
public:
    virtual unsigned int getUniqueID() = 0;
};

template<typename ClassName>
class ControllerEvent : public IControllerEvent {
private:
    static unsigned int uniqueID;
public:
    ControllerEvent() {
        if(this->uniqueID == 0)
            this->uniqueID = ++nextID;
    }

    ~ControllerEvent() {};

    static unsigned int getStaticUniqueID() {
        if(uniqueID == 0)
            uniqueID = ++nextID;
        return uniqueID;
    }

    unsigned int getUniqueID() {
        return this->uniqueID;
    }
};

template<typename ClassName>
unsigned int ControllerEvent<ClassName>::uniqueID = 0;