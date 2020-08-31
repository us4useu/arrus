addpath('C:\Users\pjarosik\arrus-releases\ref-115\matlab');
addpath('C:\Users\pjarosik\src\arrus\arrus\api\matlab');

import arrus.devices.us4r.*;
import arrus.devices.probe.*;
import arrus.session.*;

arrus.setConsoleLogger('TRACE');

rxSettings = RxSettings(...
    'dtgcAttenuation', 42,...
    'pgaGain', 24, ...
    'lnaGain', 24, ...
    'tgcSamples', [14, 15, 16], ...
    'lpfCutoff', 10e6, ...
    'activeTermination', 200 ...
);

adapterMapping =  cat(1, cat(2, zeros(1, 64), ones(1, 64)), cat(2, 0:63, 0:63));

adapterSettings = ProbeAdapterSettings(ProbeAdapterModelId('esaote2', 'us4us'), ...
    128, adapterMapping);

probeModel = ProbeModel(ProbeModelId('sl1543', 'esaote'), 128, 0.3e-3, [1e6, 10e6]);

probeSettings = ProbeSettings(probeModel, 0:127);
us4RSettings = Us4RSettings(adapterSettings, probeSettings, rxSettings);
sessionSettings = SessionSettings(us4RSettings);

Session(sessionSettings);