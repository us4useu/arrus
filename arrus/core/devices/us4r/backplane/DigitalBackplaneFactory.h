#ifndef ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORY_H
#define ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORY_H

namespace arrus::devices {

class DigitalBackplaneFactory {
public:
    virtual std::optional<DigitalBackplane::Handle> getDigitalBackplane(const HVSettings &settings,
                                                                        const std::vector<IUs4OEM *> &master) = 0;
    virtual ~DigitalBackplaneFactory() = default;
};
}

#endif//ARRUS_CORE_DEVICES_US4R_BACKPLANE_DIGITALBACKPLANEFACTORY_H
