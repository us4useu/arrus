#include "Us4OEMImpl.h"

namespace arrus {

Us4OEMImpl::Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
                       BitMask activeChannelGroups,
                       std::vector<uint8_t> channelMapping)
    : Us4OEM(id), logger{getLoggerFactory()->getLogger()},
      ius4oem(std::move(ius4oem)),
      activeChannelGroups(std::move(activeChannelGroups)),
      channelMapping(std::move(channelMapping)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}

Us4OEMImpl::~Us4OEMImpl() {
    try {
        logger->log(LogSeverity::DEBUG,
                    arrus::format("Destroying {} instance",
                                  getDeviceId().toString()));
    } catch(std::exception &e) {
        std::cerr <<
                  arrus::format("Exception while calling us4oem destructor: {}",
                                e.what())
                  << std::endl;
    }
}

}
