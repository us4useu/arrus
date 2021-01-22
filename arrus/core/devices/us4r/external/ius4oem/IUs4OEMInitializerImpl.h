#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H

#include "IUs4OEMInitializer.h"

namespace arrus::devices {

class IUs4OEMInitializerImpl : public IUs4OEMInitializer {
public:

    void initModules(std::vector<IUs4OEMHandle> &ius4oems) override {
        // Reorder us4oems according to ids (us4oem with the lowest id is the
        // first one, with the highest id - the last one).
        // TODO(pjarosik) make the below sorting exception safe
        // (currently will std::terminate on an exception).
        std::sort(std::begin(ius4oems), std::end(ius4oems),
                  [](const IUs4OEMHandle &x, const IUs4OEMHandle &y) {
                      return x->GetID() < y->GetID();
                  });

        for(auto &u : ius4oems) {
            u->Initialize(1);
        }
        // Perform successive initialization levels.
        for(int level = 2; level <= 4; level++) {
            ius4oems[0]->Synchronize();
            for(auto &u : ius4oems) {
                u->Initialize(level);
            }
        }
        // Us4OEMs are initialized here.
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMINITIALIZERIMPL_H
