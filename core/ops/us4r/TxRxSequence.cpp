#include "TxRxSequence.h"
#include "arrus/common/format.h"

namespace arrus::ops::us4r {

std::ostream &
operator<<(std::ostream &os, const TxRxSequence &seq) {
    os << "tx/rx sequence: ";
    os << "pri: " << seq.getPri();
    os << ", operations: ";
    int i = 0;
    for(auto const&[tx, rx]: seq.getOps()) {
        os << i << ": ";
        os << "TX: ";
        os << "aperture: " << toString(tx.getAperture())
           << ", delays: " << toString(tx.getDelays())
           << ", center frequency: " << tx.getPulse().getCenterFrequency()
           << ", n. periods: " << tx.getPulse().getNPeriods()
           << ", inverse: " << tx.getPulse().isInverse();
        os << "; RX: ";
        os << "aperture: " << toString(rx.getAperture());
        os << "sample range: " << rx.getSampleRange().start() << ", "
           << rx.getSampleRange().end();
        os << "fs divider: " << rx.getFsDivider();
        os << std::endl;
        ++i;
    }
    return os;
}

}