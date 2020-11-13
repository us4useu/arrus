#ifndef ARRUS_CORE_API_DEVICES_TRIGGERGENERATOR_H
#define ARRUS_CORE_API_DEVICES_TRIGGERGENERATOR_H

namespace arrus::devices {

/**
 * A device that has ability to generate triggers.
 */
class TriggerGenerator {
public:
    virtual void startTrigger() = 0;
    virtual void stopTrigger() = 0;
};

}

#endif //ARRUS_CORE_API_DEVICES_TRIGGERGENERATOR_H
