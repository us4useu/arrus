#include "RxSettings.h"

#include "arrus/common/format.h"

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const RxSettings &settings) {
    os << "dtgcAttenuation: " << ::arrus::toString(settings.getDtgcAttenuation())
       << " pgaGain: " << settings.getPgaGain()
       << " lnaGain: " << settings.getLnaGain()
       << " tgcSamples: " << ::arrus::toString(settings.getTgcSamples())
       << " lpfCutoff: " << settings.getLpfCutoff()
       << " activeTermination: " << ::arrus::toString(settings.getActiveTermination());
    return os;
}

}

