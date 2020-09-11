#include "RxSettings.h"

#include "arrus/common/format.h"

namespace arrus::devices {

std::ostream &
operator<<(std::ostream &os, const RxSettings &settings) {
    os << "dtgcAttenuation: " << toString(settings.getDTGCAttenuation())
       << " pgaGain: " << settings.getPGAGain()
       << " lnaGain: " << settings.getLNAGain()
       << " tgcSamples: " << toString(settings.getTGCSamples())
       << " lpfCutoff: " << settings.getLPFCutoff()
       << " activeTermination: " << toString(settings.getActiveTermination());
    return os;
}

}

