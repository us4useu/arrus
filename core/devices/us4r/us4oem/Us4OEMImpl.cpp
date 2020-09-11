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
        logger->log(LogSeverity::DEBUG, arrus::format("Destroying handle"));
    } catch(const std::exception &e) {
        std::cerr <<
                  arrus::format("Exception while calling us4oem destructor: {}",
                                e.what())
                  << std::endl;
        throw e;
    }
}

void Us4OEMImpl::setTxRxSequence(const ::arrus::ops::us4r::TxRxSequence &seq) {
    // TODO validate
    // tx aperture size: 128
    // rx aperture size: 128, number of active elements <= 32
    // TGC: tgc is supported only for lna + pga = 54 (check us4r.m)
    // TODO initialize module: reset all parameters
    logger->log(LogSeverity::TRACE,
                arrus::format("Setting {}", toString(seq)));

    // General sequence parameters.
    ius4oem->SetNTriggers(seq.getOps().size());
    // set tx aperture (vector -> bitset)
    // set rx aperture (vector -> bitset)
    // set active elements (based on tx and rx aperture)

    // set tx properties
    // set rx properties
    // - convert number of samples to rxTime
    // - set rxdelay to 0
    // - set scheduleReceive
    // - set rx mapping
    // -- turn off selected channels if some of them are pointing to the same physical channel
    // TGC
    // - remap to characteristic
    // - convert to specific scale
}

}
