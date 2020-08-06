#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H

#include "arrus/core/common/validation.h"
#include "arrus/core/common/logging/Logger.h"
#include "arrus/core/devices/us4oem/Us4OEMSettings.h"

#include "arrus/core/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/external/ius4oem/LNAGainValueMap.h"
#include "arrus/core/external/ius4oem/LPFCutoffValueMap.h"
#include "arrus/core/external/ius4oem/DTGCAttenuationValueMap.h"
#include "arrus/core/external/ius4oem/ActiveTerminationValueMap.h"

namespace arrus {

class Us4OEMSettingsValidator : public Validator<Us4OEMSettings> {
public:
    using Validator<Us4OEMSettings>::Validator;

    void validate(const Us4OEMSettings &obj) override {
        // Channel mappings
        // check the size
        // make sure the values are in range [0, nchannels]
        // make sure that the set of values has the same as the provided vector
        // nie wszystkie kanaly musza byc zmapowane, ale:

        // liczba elementow w mapowaniu musi byc mniejsza od liczby kanalow modulu
        // mapowanie musi dotyczyc wszystkich kanalow (nie dopuszczamy sytuacji, gdy mapowanie nie jest znane - w ten sposob mamy pewnosc, ze wszystkie kanaly sa jeden w jeden z kanalami modulu (i ze np. jeden kanal jest obslugiwany przez dwa te same kanaly wejsciowe)
        // tak wlasciwie jest to permutacja kanalow
        // uzytkownik musi podac mapowanie wszystkich kanalow
        // domyslnie kanaly niemapowane sa jeden w jeden: [1, 2, 3, 4, 5, 6, ..]
        // mapowanie adaptera nadpisuje to mapowanie jeden w jeden:
        // adapter

        // wymagania na mapowania:
        //
        // [
        // 1. kazdy kanal docelowy musi byc aktywny - mapowanie do kanalow nieaktywnych nie ma sensu
        // - np. dla glowicy esaote
        // 2. mapowanie musi byc w obrebie 32 kanalow - luzne ograniczenie na obecne potrzeby
        // 3.

        // grupy aktywnych kanalow determinuja ktore kanaly przemapowujemy
        // czyli np.

        // Active groups of channels.
        // Here check only the size of input array (should be equal Nchannels/8).
        // No additional assumptions can be made.

        // TGC values
//        Logger::log(LogSeverity::INFO, "Hello world");
//        if(obj.getDTGCAttenuation().has_value()) {
//            expectOneOf(obj.getDTGCAttenuation().value(),
//                    DTGCAttenuationValueMap::getInstance().getAvailableValues(),
//                    "dtgc attenuation"
//                    );
//        }
//        expectOneOf(obj.getPGAGain(),
//                    PGAGainValueMap::getInstance().getAvailableValues(),
//                    "pga gain");
//        expectOneOf(obj.getLNAGain(),
//                    LNAGainValueMap::getInstance().getAvailableValues(),
//                    "lna gain");
//
//        if(obj.getTGCSamples().has_value()) {
//            auto tgcMax = float(obj.getPGAGain() + obj.getLNAGain());
//            auto tgcMin = float(tgcMax - 40);
//            for(auto value : obj.getTGCSamples().value()) {
//                expectInRange(value, tgcMin, tgcMax, "tgc sample");
//            }
//        }
//
//        // Active termination.
//        expectOneOf(obj.getActiveTermination(),
//                ActiveTerminationValueMap::getInstance().getAvailableValues(),
//                "active termination");
//
//        // LPF cutoff.
//        expectOneOf(obj.getLPFCutoff(),
//                LPFCutoffValueMap::getInstance().getAvailableValues(),
//                "lpf cutoff");
    }

};

}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMSETTINGSVALIDATOR_H
